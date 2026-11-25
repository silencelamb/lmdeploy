"""Summarize R11 worker and HTTP traces.

Each step retains its critical rank.
"""

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(
        n=a.size,
        mean=float(a.mean()),
        p50=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        maximum=float(a.max()),
    )


def read(path):
    return json.loads(path.read_text())


def device(path):
    result = {}
    reference = [read(path / f'standard_rank{r}_checks.json') for r in range(8)]
    for name in ('standard', 'r11_1', 'r11_2', 'r11_3', 'r11_host_replay'):
        times = [read(path / f'{name}_rank{r}_whole_step.json') for r in range(8)]
        checks = [read(path / f'{name}_rank{r}_checks.json') for r in range(8)]
        for c, ref, tm in zip(checks, reference, times):
            assert tm['completed'] == 512 and tm['device_error'] == 0 and c['finite']
            assert all(c[key] == ref[key] for key in ('tokens', 'logits_sha256', 'state_sha256', 'pool_sha256'))
        critical = np.max([rank['step_body_ms'] for rank in times], axis=0)
        result[name] = dict(
            body_ms=stats(critical),
            loop_ms=max(r['gpu_loop_ms'] for r in times) / 512,
            first32_ms=float(critical[:32].mean()),
            last32_ms=float(critical[-32:].mean()),
            graph_nodes=checks[0].get('graph_nodes'),
            bits_equal_to_stock=True,
            frozen_tokens_equal=all(r.get('frozen_tokens_equal') for r in checks),
        )
    return result


def serving(path, startup_samples=0):
    ranks = []
    for rank in range(8):
        warm = path / f'warmup-rank{rank}.json'
        skip = read(warm) if warm.exists() else startup_samples
        ranks.append(read(path / f'clean-rank{rank}.json')[skip:])
    assert len(set(map(len, ranks))) == 1 and len(ranks[0]) > 32
    times = np.asarray(ranks, dtype=np.int64)
    periods = np.diff(times[:, :, 0], axis=1) / 1e6
    body = (times[:, :-1, 1] - times[:, :-1, 0]) / 1e6
    assert np.all(periods >= body) and np.all(body > 0)
    # A cold first capture and cohort changes at the edges remain in the all-steps result.
    rank = np.argmax(periods, axis=0)
    ix = np.arange(periods.shape[1])
    critical, on_rank = periods[rank, ix], body[rank, ix]
    steady = slice(8, -8)
    with gzip.open(path / 'clean-http.json.gz', 'rt') as f:
        reqs = json.load(f)
    assert len(reqs) == 64 and all(r['prompt_tokens'] == 65536 and len(r['tokens']) == 512 for r in reqs)
    tpot, itl = [], []
    for r in reqs:
        arrivals = r['arrivals']
        tpot.append((arrivals[-1][0] - arrivals[0][0]) * 1000 / (len(r['tokens']) - arrivals[0][1]))
        itl.extend((b[0] - a[0]) * 1000 for a, b in zip(arrivals, arrivals[1:]) if a[1] == b[1] == 1)
    return dict(
        requests=64,
        input_tokens=64 * 65536,
        output_tokens=64 * 512,
        selected_steps=len(ranks[0]),
        all_start_to_start_ms=stats(critical),
        steady_start_to_start_ms=stats(critical[steady]),
        steady_body_on_critical_rank_ms=stats(on_rank[steady]),
        steady_gap_on_same_rank_ms=stats((critical - on_rank)[steady]),
        steady_max_body_ms=stats(body.max(axis=0)[steady]),
        request_tpot_ms=stats(tpot),
        single_token_sse_itl_ms=stats(itl),
    )


def union_duration(events, begin, end):
    previous = begin
    total = 0.0
    for event in events:
        t = event['ts']
        if t >= end:
            break
        stop = min(t + event['dur'], end)
        start = max(t, begin, previous)
        if stop > start:
            total += stop - start
            previous = stop
    return total


def trace(path):
    with gzip.open(path / 'worker-trace.json.gz', 'rt') as f:
        events = json.load(f)['traceEvents']
    gpu = sorted(
        (e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('dur', 0) > 0),
        key=lambda e: e['ts'],
    )
    starts = [e['ts'] for e in gpu if '_r11_body_start' in e['name']]
    ends = [e['ts'] + e['dur'] for e in gpu if '_r11_body_end' in e['name']]
    periods, bodies, gaps, busy = [], [], [], []
    names = collections.Counter()
    for left, right in zip(starts, starts[1:]):
        boundary = [t for t in ends if left < t < right]
        if len(boundary) != 1:
            continue
        (stop,) = boundary
        periods.append((right - left) / 1000)
        bodies.append((stop - left) / 1000)
        gaps.append((right - stop) / 1000)
        busy.append(union_duration(gpu, stop, right) / 1000)
        for e in gpu:
            if stop <= e['ts'] < right:
                names[e['name']] += e['dur'] / 1000
    if not periods:
        raise ValueError('No full R11 graph intervals in the CUPTI window')
    return dict(
        rank=0,
        periods_ms=stats(periods),
        body_ms=stats(bodies),
        outside_ms=stats(gaps),
        outside_device_busy_union_ms=stats(busy),
        outside_no_cuda_activity_ms=stats(np.subtract(gaps, busy)),
        outside_top_gpu_ops_ms_per_step=[(k, v / len(periods)) for k, v in names.most_common(12)],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=Path)
    parser.add_argument('--serving', type=Path)
    parser.add_argument('--startup-samples', type=int, default=0, help='Synthetic samples in legacy timer dumps')
    args = parser.parse_args()
    data = {}
    if args.device:
        data['device'] = device(args.device)
    if args.serving:
        data['serving'] = serving(args.serving, args.startup_samples)
        data['cupti'] = trace(args.serving)
    print(json.dumps(data, indent=2))
