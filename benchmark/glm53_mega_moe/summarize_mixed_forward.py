"""Check a genuinely mixed forward and compare two complete AgentX exports at the same load."""

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import quantiles

from compare_agentx_scheduling import paired, read_profile

TARGET = re.compile(r'Mixed target forward rank=(\d+) count=(\d+) prefill_tokens=(\d+) decode_rows=(\d+) '
                    r'verify_width=(\d+)')
ACCEPTED = re.compile(r'Mixed verify accepted histogram=(\[[0-9, ]+\])')


def target_evidence(path, budget=8192):
    highest = {}
    widths = set()
    tokens = []
    accepted = Counter()
    with path.open(errors='replace') as log:
        for line in log:
            if match := TARGET.search(line):
                rank, count, prompt, decode, width = map(int, match.groups())
                assert prompt > 0 and decode > 0 and width in (1, 4)
                total = prompt + decode * width
                assert total <= budget
                highest[rank] = max(highest.get(rank, 0), count)
                widths.add(width)
                tokens.append((prompt, total))
            if match := ACCEPTED.search(line):
                bins = ast.literal_eval(match.group(1))
                assert len(bins) == 4 and all(n >= 0 for n in bins)
                accepted.update(dict(enumerate(bins)))
    if not tokens:
        raise ValueError(f'No mixed target model execution recorded in {path}')
    return dict(logged_steps=len(tokens), worker_ranks=sorted(highest), verify_widths=sorted(widths),
                mixed_steps_lower_bound=sum(highest.values()), max_prompt_tokens=max(p for p, _ in tokens),
                max_total_query_tokens=max(t for _, t in tokens), logged_accepted_draft_counts=dict(accepted))


def tail(rows):
    def p95(vals):
        if len(vals) == 1:
            return vals[0]
        return quantiles(vals, n=100, method='inclusive')[94]

    result = paired(rows)
    if rows:
        for field in ('ttft', 'tpot'):
            for idx, arm in enumerate(('reference', 'candidate')):
                result[f'{arm}_{field}_p95_ms'] = round(p95([pair[idx][field] for pair in rows]), 3)
    return result


def matched_tail_interval(reference, before, after, repeats=1000):
    """Resample whole Weka source trajectories among turns scored on both sides."""
    keys = sorted(before.keys() & after.keys(), key=str)
    index = {key: i for i, key in enumerate(keys)}
    groups = defaultdict(list)
    with (reference / 'profile_export.jsonl').open() as stream:
        for line in stream:
            meta = json.loads(line)['metadata']
            if meta['benchmark_phase'] != 'profiling':
                continue
            key = tuple(meta.get(k) for k in ('conversation_id', 'turn_index', 'source_outer_idx', 'source_inner_idx'))
            if key in index:
                groups[meta.get('source_trace_id') or key[0]].append(index[key])
    units = list(groups.values())
    if len(units) < 2:
        return None
    assert sum(map(len, units)) == len(keys)
    times = [(before[k]['ttft'], after[k]['ttft'], before[k]['tpot'], after[k]['tpot']) for k in keys]
    rng = Random(20260707)
    draws = []
    for _ in range(repeats):
        sample = [times[i] for unit in rng.choices(units, k=len(units)) for i in unit]
        ref_ttft, cand_ttft, ref_tpot, cand_tpot = [quantiles(col, n=100, method='inclusive') for col in zip(*sample)]
        draws.append((cand_ttft[94] - ref_ttft[94], cand_tpot[49] - ref_tpot[49], cand_tpot[94] - ref_tpot[94]))
    bounds = [quantiles(col, n=40, method='inclusive') for col in zip(*draws)]
    intervals = [[round(quant[0], 3), round(quant[-1], 3)] for quant in bounds]
    return dict(source_traces=len(units), turns=len(keys), resamples=repeats, seed=20260707,
                delta_ms=dict(zip(('ttft_p95', 'tpot_p50', 'tpot_p95'), intervals)))


def agentx_delta(reference, candidate):
    original_setup, original, before = read_profile(reference)
    new_setup, changed, after = read_profile(candidate)
    if original_setup != new_setup:
        raise ValueError(f'Different QA replay conditions: {original_setup} vs {new_setup}')
    for result, path in ((original, reference), (changed, candidate)):
        report = json.loads((path / 'profile_export_aiperf.json').read_text())
        result['ttft_p95_ms'] = report['time_to_first_token']['p95']
        result['tpot_p95_ms'] = report['inter_token_latency']['p95']
        result['tpot_p99_ms'] = report['inter_token_latency']['p99']
        result['cached_input_fraction'] = (report['total_usage_prompt_cache_read_tokens']['avg']
                                           / report['total_usage_prompt_tokens']['avg'])
    common = [(before[key], after[key]) for key in before.keys() & after.keys()]
    same_work = [(a, b) for a, b in common if (a['shape'], a['cached']) == (b['shape'], b['cached'])]
    buckets = {}
    for label, low, high in (('under_4k', 0, 4096), ('4k_to_32k', 4096, 32768), ('32k_and_above', 32768, 2**63)):
        buckets[label] = tail([(a, b) for a, b in common if low <= a['shape'][0] - a['cached'] < high])
    delta = {key: round(100 * (changed[key] / original[key] - 1), 3) for key in
             ('ttft_p95_ms', 'tpot_p50_ms', 'tpot_p95_ms', 'output_tokens_per_s', 'goodput_rps')}
    return dict(reference=original, candidate=changed, change_percent=delta, common_turns=tail(common),
                common_equal_input_output_cache=tail(same_work), common_by_reference_uncached_tokens=buckets,
                matched_source_bootstrap_95=matched_tail_interval(reference, before, after))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-log', type=Path, help='Worker log containing mixed target forward counters.')
    parser.add_argument('--reference', type=Path, help='Reference AIPerf export directory.')
    parser.add_argument('--candidate', type=Path, help='Candidate AIPerf export directory.')
    parser.add_argument('--budget', type=int, default=8192, help='Maximum query tokens per rank and target forward.')
    args = parser.parse_args()
    if bool(args.reference) != bool(args.candidate) or not (args.server_log or args.reference):
        parser.error('Supply a worker log, two matching AgentX exports, or both.')
    result = {}
    if args.server_log:
        result['worker_evidence'] = target_evidence(args.server_log, args.budget)
    if args.reference:
        result['c90'] = agentx_delta(args.reference, args.candidate)
    print(json.dumps(result, indent=2, allow_nan=False))
