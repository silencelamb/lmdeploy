"""Rebuild scheduling A/B metrics from completed fixed-I/O and interference runs."""

import argparse
import gzip
import json
import re
import shlex
from pathlib import Path
from statistics import median, quantiles

PHASES = ('request_queue_time_seconds', 'request_prefill_time_seconds', 'request_decode_time_seconds')
SCHEDULER_ENV = {'LMDEPLOY_INTERLEAVED_PREFILL_TOKENS', 'LMDEPLOY_INTERLEAVED_DECODE_ROUNDS',
                 'LMDEPLOY_DP_PREFILL_PERIOD', 'LMDEPLOY_DP_PREFILL_TRACE'}
DP_TRACE = re.compile(r'DP admission round=(\d+) time=([\d.]+) prefill_ranks=(\d+)')


def read_json(path):
    return json.loads(path.read_text())


def streams(path, count, n_input, n_output):
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == count and {row['index'] for row in rows} == set(range(count)), path
    for row in rows:
        arrivals = row['arrivals_s']
        assert (not row.get('error') and row['input_tokens'] == n_input
                and row['output_tokens'] == len(row['output_ids']) == n_output), path
        assert sum(size for _, size in arrivals) == n_output and all(size > 0 for _, size in arrivals), path
        assert all(a[0] <= b[0] for a, b in zip(arrivals, arrivals[1:])), path
    return sorted(rows, key=lambda row: row['index'])


def tpot(row):
    arrivals = row['arrivals_s']
    return 1000 * (arrivals[-1][0] - arrivals[0][0]) / (row['output_tokens'] - 1)


def percentiles(values):
    percentiles = quantiles(values, n=100, method='inclusive')
    return {f'p{q}': round(percentiles[q - 1], 3) for q in (50, 95, 99)}


def summarize_probes(fixed, repeats, depth):
    arms = {}
    metrics = {}
    for arm in ('alone', 'with-prefill'):
        probes = []
        extra_wave_seconds = []
        for rep in range(repeats):
            tag = f'probe32-r{rep}-{arm}'
            batch = streams(fixed / f'{tag}-probes.jsonl.gz', 32, 1024, 512)
            meta = read_json(fixed / f'{tag}-summary.json')
            marker = meta['injection_marker_s']
            assert 0 < meta['spec']['proposed_per_draft'] <= depth
            assert max(row['started_s'] + row['arrivals_s'][0][0] for row in batch) < marker
            assert marker < min(row['finished_s'] for row in batch)
            if arm == 'with-prefill':
                extra = streams(fixed / f'{tag}-prefills.jsonl.gz', 24, 8192, 1)
                first = min(row['started_s'] for row in extra)
                assert first >= marker
                extra_wave_seconds.append(round(max(row['finished_s'] for row in extra) - first, 3))
            probes.extend(batch)
        arms[arm] = probes
        gaps = [1000 * max(b[0] - a[0] for a, b in zip(r['arrivals_s'], r['arrivals_s'][1:])) for r in probes]
        metrics[arm] = dict(requests=len(probes), tpot_ms=percentiles([tpot(r) for r in probes]),
                            max_inter_chunk_gap_ms=percentiles(gaps))
        if extra_wave_seconds:
            metrics[arm]['extra_prefill_wave_s'] = extra_wave_seconds
    paired = list(zip(arms['alone'], arms['with-prefill']))
    assert all(a['prompt_sha256'] == b['prompt_sha256'] for a, b in paired)
    identical = [(a, b) for a, b in paired if a['output_ids'] == b['output_ids']]
    metrics['identical_512_ids_within_pair'] = len(identical)
    if identical:
        metrics['median_tpot_penalty_identical_ids_ms'] = round(median(tpot(b) - tpot(a) for a, b in identical), 3)
    return metrics


def summarize(run):
    assert (run / 'exit_code').read_text().strip() == '0', run
    settings = run / 'server.env'
    overrides = {}
    if settings.exists():
        for line in settings.read_text().splitlines():
            key, _, value = line.partition('=')
            if key in SCHEDULER_ENV:
                overrides[key] = value
    fixed = run / 'fixed'
    server = (run / 'server-command.sh').read_text()
    if match := re.search(r'GLM53_SPEC_TOKENS=(\d+)', server):
        backend, depth = 'lmdeploy', int(match.group(1))
    else:
        argv = shlex.split(server)
        backend = 'vllm'
        depth = json.loads(argv[argv.index('--speculative-config') + 1])['num_speculative_tokens']
    if depth:
        workload = read_json(fixed / 'workload.json')
        assert workload['draft_depth'] == depth and workload['workers'] == 8
    cases = sorted(fixed.glob('i8192-o256-c64-r*'))
    assert len(cases) == (2 if depth else 1), run
    all_rows, rates, seeds, windows = [], [], [], []
    elapsed = 0.0
    drafts = proposed = accepted = 0
    sums = dict.fromkeys(PHASES, 0.0)
    for rep, case in enumerate(cases):
        summary = read_json(case / 'summary.json')
        assert summary['requests'] == summary['success'] == 256 and summary['placement'] == 'least-loaded'
        rows = streams(case / 'requests.jsonl.gz', 256, 8192, 256)
        all_rows.extend(rows)
        seconds = max(r['finished_s'] for r in rows) - min(r['started_s'] for r in rows)
        assert abs(seconds - summary['wall_s']) < 1.e-6
        elapsed += seconds
        rates.append(round(len(rows) * 256 / seconds, 2))
        seeds.append(summary['seed'])
        if depth:
            stat = read_json(fixed / f'rolling-r{rep}-counters.json')
            assert stat['time_to_first_token_seconds_count'] == len(rows) and 0 < stat['proposed_per_draft'] <= depth
            drafts += stat['spec_decode_num_drafts_total']
            proposed += stat['spec_decode_num_draft_tokens_total']
            accepted += stat['spec_decode_num_accepted_tokens_total']
            for key in PHASES:
                sums[key] += stat[key + '_sum']
            before, after = [fixed / f'rolling-r{rep}-{tag}-worker-counters.json' for tag in ('before', 'after')]
        else:
            stat = next(r for r in read_json(fixed / 'phase-summary.json') if r['shape'] == 'i8192-o256-c64')
            assert stat['stream_valid'] and stat['requests'] == len(rows)
            for key in PHASES:
                assert stat['phases'][key]['count'] == len(rows)
                sums[key] += stat['phases'][key]['sum_s']
            before, after = [fixed / f'i8192-o256-c64-{tag}-rank0.prom' for tag in ('before', 'after')]
        windows.append((before.stat().st_mtime, after.stat().st_mtime))
    count = len(all_rows)
    result = dict(backend=backend, draft_depth=depth, env=overrides, requests=count, seeds=seeds,
                  output_tok_s=round(count * 256 / elapsed, 2), output_tok_s_by_repeat=rates,
                  ttft_ms=percentiles([1000 * r['arrivals_s'][0][0] for r in all_rows]),
                  tpot_ms=percentiles([tpot(r) for r in all_rows]),
                  e2e_ms=percentiles([1000 * (r['finished_s'] - r['started_s']) for r in all_rows]),
                  phase_means_s={k: round(v / count, 4) for k, v in sums.items()})
    if depth:
        assert 0 < proposed <= depth * drafts and 0 <= accepted <= proposed
        if backend == 'lmdeploy':
            assert proposed == depth * drafts
        result['mean_tokens_per_verification'] = round(1 + accepted / drafts, 4)
        result['probes'] = summarize_probes(fixed, len(cases), depth)
    if overrides.get('LMDEPLOY_DP_PREFILL_TRACE') == '1':
        with (run / 'server.log').open() as log:
            trace = [m.groups() for line in log if (m := DP_TRACE.search(line))]
        ranks = [int(n) for _, t, n in trace if int(n) and any(lo <= float(t) <= hi for lo, hi in windows)]
        assert ranks and all(1 <= n <= 8 for n in ranks)
        result['dp_trace'] = dict(prefill_rounds=len(ranks), rank_prefills=sum(ranks),
                                  mean_prefill_ranks=round(sum(ranks) / len(ranks), 3))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps({run.name: summarize(run) for run in args.runs}, indent=2))
