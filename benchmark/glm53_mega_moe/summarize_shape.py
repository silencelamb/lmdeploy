"""Aggregate the synchronized rank results and the rank-0 CUDA kernel trace."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def kernel_category(name):
    if 'sm90_fp8_fp4_mega_moe' in name or 'Sm90MegaMoE' in name:
        return 'fused_moe'
    if '_stage_mega_moe' in name:
        return 'staging'
    if 'deep_ep::' in name:
        if '::dispatch<' in name:
            return 'deepep_dispatch'
        if '::combine<' in name:
            return 'deepep_combine'
    if '_grouped_fp8_gemm' in name or ('sm90_fp8_gemm' in name and 'MGrouped' in name):
        return 'grouped_expert_gemm'
    if 'silu_and_mul' in name:
        return 'swiglu'
    if 'topk_reduce' in name:
        return 'separate_combine_reduce'
    if 'fp8_gemm' in name:
        return 'other_fp8_gemm'
    return 'other_cuda_kernels'


def trace_summary(path):
    with gzip.open(path, 'rt') as profile:
        events = json.load(profile)['traceEvents']
    steps = sum(event.get('cat') == 'user_annotation' and event.get('name') == 'glm53_decode_step'
                for event in events)
    counts = Counter()
    microseconds = Counter()
    for event in events:
        if event.get('cat') == 'kernel':
            category = kernel_category(event['name'])
            counts[category] += 1
            microseconds[category] += event['dur']
    return dict(steps=steps, calls=dict(counts),
                kernel_ms_per_step={category: total / (1000 * steps) for category, total in microseconds.items()})


def summarize(path):
    ranks = [json.loads((path / f'rank{rank}.json').read_text()) for rank in range(8)]
    repeats = list(zip(*(row['repeats'] for row in ranks), strict=True))
    result = dict(
        backend=ranks[0]['backend'], batch=ranks[0]['batch'], context=ranks[0]['context'], eager=ranks[0]['eager'],
        kv_blocks=ranks[0]['gpu_blocks'], kv_bytes=ranks[0]['kv_bytes'],
        prefill_s=median(r['prefill_s'] for r in ranks),
        max_rank_gpu_ms=[max(r['mean_device_ms'] for r in run) for run in repeats],
        max_rank_wall_ms=[max(mean(r['wall_ms']) for r in run) for run in repeats],
        peak_reserved_gib=max(r['peak_reserved_bytes'] for run in repeats for r in run) / (1 << 30),
        min_repeat_token_agreement=min(r['repeat_argmax_agreement'] for run in repeats for r in run),
        graph_count=[row['graph_count'] for row in ranks],
    )
    if 'accuracy' in ranks[0]:
        accuracy = [row['accuracy'] for row in ranks]
        # The leading legacy samples are the uint8 allocation pools used for
        # exact storage checks. Only the following samples are typed model states.
        kv_pools = ranks[0]['kv_pools']
        result.update(
            prefill_token_match=all(row['prefill_exact'] for row in accuracy),
            prefill_state_max_abs=max(s['max_abs'] for row in accuracy for s in row['prefill_states'][kv_pools:]),
            decode_token_agreement=mean(row['argmax_agreement'] for row in accuracy),
            logits_max_relative_l2=max(row['logits']['relative_l2'] for row in accuracy),
            state_max_relative_l2=max(state['relative_l2'] for row in accuracy for state in row['states'][kv_pools:]),
        )
        if 'prefill_token_matches' in accuracy[0]:
            result['prefill_token_matches'] = sum(row['prefill_token_matches'] for row in accuracy)
            result['prefill_token_count'] = sum(rank['batch'] for rank in ranks)
        for phase in ('prefill_kv', 'decode_kv'):
            if phase in accuracy[0]:
                result[f'{phase}_max_relative_l2'] = {
                    key: max(row[phase][key]['relative_l2'] for row in accuracy) for key in accuracy[0][phase]}
    trace = path / 'profile-rank0.json.gz'
    if trace.is_file():
        result['rank0_profile'] = trace_summary(trace)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps({path.name: summarize(path) for path in args.runs}, indent=2))
