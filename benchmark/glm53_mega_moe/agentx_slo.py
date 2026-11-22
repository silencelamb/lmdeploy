"""Score one completed AgentX MVP window and decide whether to run the next concurrency."""
import argparse
import json
import math
from pathlib import Path


def score(export_path: Path, duration_s: int = 1800, ttft_limit_ms: float = 3000, tpot_limit_ms: float = 25) -> dict:
    report = json.loads(export_path.read_text())
    meta = report['metadata']
    if meta['submission_valid'] is not True or meta['scenario'] != 'inferencex-agentx-mvp':
        raise ValueError('An eligible AgentX MVP submission is required.')
    windows = [w for w in meta['metric_duration_coverage'] if w['phase_name'] == 'profiling']
    if len(windows) != 1 or windows[0]['expected_duration_seconds'] != duration_s:
        raise ValueError(f'Expected one {duration_s}-second profiling window.')
    ttft = report['time_to_first_token']
    tpot = report['inter_token_latency']
    if ttft['unit'] != 'ms' or tpot['unit'] != 'ms':
        raise ValueError('The exported TTFT and per-request ITL metrics must be in milliseconds.')
    ttft_p50, tpot_p50 = ttft['p50'], tpot['p50']
    completed = report['request_count']['avg']
    good = report['good_request_count']['avg']
    if completed <= 0 or not 0 <= good <= completed or not all(map(math.isfinite, (ttft_p50, tpot_p50))):
        raise ValueError('Missing usable request counts or median latency.')
    return dict(
        formal_seconds=duration_s,
        observed_seconds=report['benchmark_duration']['avg'],
        request_count=completed,
        good_requests=good,
        goodput_rps=report['goodput']['avg'],
        attainment=good / completed,
        ttft_p50_ms=ttft_p50,
        tpot_p50_ms=tpot_p50,
        user_decode_p50_tps=report['output_token_throughput_per_user']['p50'],
        output_tokens_per_s=report['output_token_throughput']['avg'],
        both_medians_pass=ttft_p50 < ttft_limit_ms and tpot_p50 < tpot_limit_ms,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path, help='Path to profile_export_aiperf.json')
    parser.add_argument('--duration', type=int, default=1800)
    parser.add_argument('--ttft-limit-ms', type=float, default=3000)
    parser.add_argument('--tpot-limit-ms', type=float, default=25)
    args = parser.parse_args()
    result = score(args.export, args.duration, args.ttft_limit_ms, args.tpot_limit_ms)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['both_medians_pass'] else 10)
