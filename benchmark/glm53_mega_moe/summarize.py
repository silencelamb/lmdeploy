"""Pool Fixed I/O samples across repetitions and read the AgentX SLO exports."""

import argparse
import csv
import gzip
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path

from fixed_io import quantiles


def load(path):
    return json.loads(path.read_text())


def fixed_io(path):
    cases = defaultdict(list)
    for summary_file in sorted(path.glob('fixed/*/summary.json')):
        summary = load(summary_file)
        key = (summary['context'], summary['output_tokens'], summary['concurrency'])
        records = summary_file.with_name('requests.jsonl.gz')
        with gzip.open(records, 'rt', encoding='utf-8') as stream:
            requests = [json.loads(line) for line in stream]
        cases[key].append((summary, requests))

    results = []
    for (context, output_tokens, concurrency), repetitions in sorted(cases.items()):
        requests = [row for _, rows in repetitions for row in rows]
        good = [row for row in requests if row['error'] is None]
        wall = sum(summary['wall_s'] for summary, _ in repetitions)
        ttf = [row['arrivals_s'][0][0] * 1000 for row in good]
        tpot = [(row['arrivals_s'][-1][0] - row['arrivals_s'][0][0]) * 1000 / (row['output_tokens'] - 1)
                for row in good if row['output_tokens'] > 1]
        itl = [(current - previous) * 1000 / count
               for row in good for (previous, _), (current, count) in zip(row['arrivals_s'], row['arrivals_s'][1:])
               for _ in range(count)]
        results.append(dict(
            context=context, output_tokens=output_tokens, concurrency=concurrency,
            repeats=len(repetitions), requests=len(requests), success=len(good),
            output_tokens_per_s=sum(row['output_tokens'] for row in good) / wall,
            ttft_ms=quantiles(ttf), tpot_ms=quantiles(tpot), itl_ms=quantiles(itl),
            end_to_end_ms=quantiles([(row['finished_s'] - row['started_s']) * 1000 for row in good]),
            peak_gpu_mib=max(summary['peak_gpu_mib'] or 0 for summary, _ in repetitions)))
    return results


def agentx(path):
    results = []
    for profile in sorted(path.glob('agentx-c*/profile_export_aiperf.json')):
        summary = load(profile)
        ttft = summary['time_to_first_token']['p50']
        tpot = summary['inter_token_latency']['p50']
        cached_tokens = summary['total_usage_prompt_cache_read_tokens']['avg']
        prompt_tokens = summary['total_usage_prompt_tokens']['avg']
        command = shlex.split(profile.with_name('command.sh').read_text())
        raw = profile.with_name('profile_export.jsonl')
        outcomes, time_buckets = agentx_outcomes(raw)
        gpu_csv = profile.with_name('gpu.csv')
        gpu_peaks = []
        if gpu_csv.exists():
            with gpu_csv.open() as source:
                samples = csv.DictReader(source, skipinitialspace=True)
                readings = (row['memory.used [MiB]'].split()[0] for row in samples)
                gpu_peaks = [int(reading) for reading in readings if reading.isdigit()]
        results.append(dict(
            concurrency=int(profile.parent.name.removeprefix('agentx-c')),
            duration_s=float(command[command.index('--benchmark-duration') + 1]),
            observed_duration_s=summary['benchmark_duration']['avg'],
            requests_per_lane_warmup=int(command[command.index('--warmup-requests-per-lane') + 1]),
            diagnostic='--unsafe-override' in command,
            request_count=summary['request_count']['avg'], good_requests=summary['good_request_count']['avg'],
            ttft_ms_p50=ttft, tpot_ms_p50=tpot, both_medians_pass=ttft < 3000 and tpot < 25,
            output_tokens_per_user_s_p50=summary['output_token_throughput_per_user']['p50'],
            ttft_ms_p95=summary['time_to_first_token']['p95'],
            tpot_ms_p95=summary['inter_token_latency']['p95'],
            itl_ms_p95=summary['inter_chunk_latency']['p95'],
            request_latency_ms_p95=summary['request_latency']['p95'],
            output_tokens_per_s=summary['output_token_throughput']['avg'],
            request_outcomes=outcomes, start_time_buckets=time_buckets,
            cached_prompt_token_ratio=cached_tokens / prompt_tokens if prompt_tokens else None,
            peak_observed_kv_cache_usage=peak_kv_usage(profile.with_name('server_metrics_export.csv')),
            peak_gpu_mib=max(gpu_peaks, default=None),
            prompt_length_buckets=agentx_lengths(raw),
            profile=str(profile)))
    return results


def agentx_outcomes(path):
    phases = defaultdict(Counter)
    completed = []
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            meta = row['metadata']
            phase = phases[meta['benchmark_phase']]
            if meta.get('was_cancelled'):
                status = 'cancelled'
            elif row.get('error'):
                status = row['error']['type']
            else:
                status = 'completed'
                if meta['benchmark_phase'] == 'profiling':
                    completed.append((meta['request_start_ns'], row['metrics']))
            phase[status] += 1
    buckets = defaultdict(list)
    if completed:
        first = min(start for start, _ in completed)
        for start, metrics in completed:
            minutes = int((start - first) // (300 * 10 ** 9)) * 5
            buckets[minutes].append({name: metric['value'] for name, metric in metrics.items() if 'value' in metric})
    windows = [dict(start_minute=start, requests=len(rows),
                    uncached_prompt_tokens=sum(r.get('usage_prompt_tokens', 0) -
                                               r.get('usage_prompt_cache_read_tokens', 0) for r in rows),
                    generated_tokens=sum(r.get('usage_completion_tokens', 0) for r in rows),
                    ttft_ms=quantiles([r['time_to_first_token'] for r in rows if 'time_to_first_token' in r]),
                    tpot_ms=quantiles([r['inter_token_latency'] for r in rows if 'inter_token_latency' in r]))
               for start, rows in sorted(buckets.items())]
    return phases, windows


def peak_kv_usage(path):
    if not path.exists():
        return None
    with path.open() as source:
        rows = csv.DictReader(line for line in source if line.strip() and not line.startswith('#'))
        samples = [float(r['max']) for r in rows if r['Metric'] == 'lmdeploy:gpu_cache_usage_perc' and r['max']]
    return max(samples, default=None)


def agentx_lengths(path):
    samples = defaultdict(list)
    boundaries = [2048, 8192, 32768, 65536, 131072, 262144]
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if (row['metadata']['benchmark_phase'] != 'profiling' or row['metadata'].get('was_cancelled')
                    or row.get('error')):
                continue
            metrics = {name: value['value'] for name, value in row['metrics'].items() if 'value' in value}
            if 'usage_prompt_tokens' not in metrics or 'time_to_first_token' not in metrics:
                continue
            length = metrics['usage_prompt_tokens']
            upper = next((b for b in boundaries if length < b), None)
            samples[upper].append(metrics)
    return [dict(input_tokens_lt=b, requests=len(rows),
                 ttft_ms=quantiles([r['time_to_first_token'] for r in rows]),
                 tpot_ms=quantiles([r['inter_token_latency'] for r in rows if 'inter_token_latency' in r]),
                 cached_prompt_tokens=sum(r.get('usage_prompt_cache_read_tokens', 0) for r in rows),
                 prompt_tokens=sum(r['usage_prompt_tokens'] for r in rows))
            for b, rows in sorted(samples.items(), key=lambda item: item[0] or float('inf'))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps({path.name: dict(fixed_io=fixed_io(path), agentx=agentx(path)) for path in args.runs}, indent=2))
