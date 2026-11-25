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


def device(path, reference_path=None):
    result = {}
    reference_path = reference_path or path
    reference = [read(reference_path / f'standard_rank{r}_checks.json') for r in range(8)]
    repeats = sorted(int(f.name.split('_')[1]) for f in path.glob('r11_[0-9]*_rank0_checks.json'))
    cases = ['standard', *(f'r11_{i}' for i in repeats)]
    if (path / 'r11_host_replay_rank0_checks.json').exists():
        cases.append('r11_host_replay')
    for name in cases:
        source = reference_path if name == 'standard' else path
        times = [read(source / f'{name}_rank{r}_whole_step.json') for r in range(8)]
        checks = [read(source / f'{name}_rank{r}_checks.json') for r in range(8)]
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
    return dict(
        **client(path),
        selected_steps=len(ranks[0]),
        all_start_to_start_ms=stats(critical),
        steady_start_to_start_ms=stats(critical[steady]),
        steady_body_on_critical_rank_ms=stats(on_rank[steady]),
        steady_gap_on_same_rank_ms=stats((critical - on_rank)[steady]),
        steady_max_body_ms=stats(body.max(axis=0)[steady]),
    )


def client(path):
    with gzip.open(path / 'clean-http.json.gz', 'rt') as f:
        reqs = json.load(f)
    assert len(reqs) == 64 and all(r['prompt_tokens'] == 65536 and len(r['tokens']) == 512 for r in reqs)
    tpot, itl = [], []
    for r in reqs:
        arrivals = r['arrivals']
        tpot.append((arrivals[-1][0] - arrivals[0][0]) * 1000 / (len(r['tokens']) - arrivals[0][1]))
        itl.extend((b[0] - a[0]) * 1000 for a, b in zip(arrivals, arrivals[1:]) if a[1] == b[1] == 1)
    return dict(
        requests=len(reqs),
        input_tokens=64 * 65536,
        output_tokens=64 * 512,
        ttft_s=stats([row['arrivals'][0][0] for row in reqs]),
        request_tpot_ms=stats(tpot),
        single_token_sse_itl_ms=stats(itl),
    )


def route_load(path, pattern, skip=0):
    counts = np.stack([np.load(path / pattern.format(rank=r))['counts'][skip:] for r in range(8)])
    assert counts.shape[2:] == (42, 54)
    assert np.all(counts >= 0) and np.all(counts.sum((0, 3)) == 64 * 8)
    # Take the heaviest rank separately for each sequential layer. M32 can
    # request an expert again if more than 32 tokens routed to it.
    unique = (counts > 0).sum(-1).max(axis=0).sum(axis=-1)
    m32 = ((counts + 31) // 32).sum(-1).max(axis=0).sum(axis=-1)
    bytes_per_expert = 3 * 4096 * 2048
    return dict(
        steps=len(unique),
        routes_per_layer=512,
        critical_unique_fp8_GB=stats(unique[8:-8] * bytes_per_expert / 1e9),
        critical_m32_fp8_GB=stats(m32[8:-8] * bytes_per_expert / 1e9),
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


def trace(path, fixed=False):
    file = 'fixed-trace.json.gz' if fixed else 'worker-trace.json.gz'
    start_name, end_name = ('start_step', 'finish_step') if fixed else ('_r11_body_start', '_r11_body_end')
    with gzip.open(path / file, 'rt') as f:
        events = json.load(f)['traceEvents']
    gpu = sorted(
        (e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('dur', 0) > 0),
        key=lambda e: e['ts'],
    )
    starts = [e['ts'] for e in gpu if start_name in e['name']]
    ends = [e['ts'] + e['dur'] for e in gpu if end_name in e['name']]
    periods, bodies, gaps, busy = [], [], [], []
    names = collections.Counter()
    experts, packets = [], []
    # The lab trace also contains its capture and one priming replay. The last
    # 30 complete intervals are inside its 32-step timed host-queued loop.
    intervals = list(zip(starts, starts[1:]))
    for left, right in intervals[-30:] if fixed else intervals:
        boundary = [t for t in ends if left < t < right]
        if len(boundary) != 1:
            continue
        (stop,) = boundary
        periods.append((right - left) / 1000)
        bodies.append((stop - left) / 1000)
        gaps.append((right - stop) / 1000)
        busy.append(union_duration(gpu, stop, right) / 1000)
        inside = [e for e in gpu if left <= e['ts'] < stop and e.get('cat') == 'kernel']
        routed = [e['dur'] / 1000 for e in inside if '::routed<0>' in e['name']]
        assert len(routed) == 42
        experts.append(routed)
        packets.append(sum(e['dur'] / 1000 for e in inside if 'prepare_mailbox' in e['name']))
        for e in gpu:
            if stop <= e['ts'] < right:
                names[e['name']] += e['dur'] / 1000
    if not periods:
        raise ValueError('No full R11 graph intervals in the CUPTI window')
    return dict(
        rank=0,
        routed_expert_ms=stats(np.sum(experts, axis=1)),
        routed_ms_by_layer=np.mean(experts, axis=0).tolist(),
        packet_prepare_ms=stats(packets),
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
    parser.add_argument('--reference', type=Path, help='Separate ordinary worker output for a full-only benchmark')
    parser.add_argument('--serving', type=Path)
    parser.add_argument('--client-only', type=Path)
    parser.add_argument('--route-dir', type=Path)
    parser.add_argument('--route-pattern', default='clean-routes-rank{rank}.npz')
    parser.add_argument('--startup-samples', type=int, default=0, help='Synthetic samples in legacy timer dumps')
    args = parser.parse_args()
    data = {}
    if args.device:
        data['device'] = device(args.device, args.reference)
        if (args.device / 'fixed-trace.json.gz').is_file():
            data['fixed_cupti'] = trace(args.device, fixed=True)
    if args.serving:
        data['serving'] = serving(args.serving, args.startup_samples)
        if (args.serving / 'worker-trace.json.gz').exists():
            data['cupti'] = trace(args.serving)
    if args.client_only:
        data['client'] = client(args.client_only)
    if args.route_dir:
        data['routing'] = route_load(args.route_dir, args.route_pattern, args.startup_samples)
    print(json.dumps(data, indent=2))
