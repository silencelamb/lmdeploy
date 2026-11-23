"""Group a full-rank GLM-5.3 decode Chrome trace by kernel family and actual target forwards."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def kernel_family(name):
    key = name.lower()
    if 'glm53_kda_output_r11' in key:
        return 'dense_matmul'
    if 'sm90_fp8_fp4_mega_moe' in key or 'sm90_fp8_fp8_mega_moe' in key:
        return 'mega_moe'
    if 'stage_mega_moe' in key or 'prepare_mega_moe' in key:
        return 'moe_staging'
    if 'noaux_tc' in key or 'single_group_topk' in key:
        return 'moe_router_topk'
    if any(s in key for s in ('mhc', 'hc_pre', 'hc_post', 'hc_project', 'hc_scale', 'hc_split')):
        return 'mhc'
    if 'nvjet' in key or 'cublaslt' in key or ('cutlass::' in key and 'gemm' in key):
        return 'dense_matmul'
    if any(s in key for s in ('kpool', 'indexer', 'deepselect', 'flashmla', 'sparse_attn', 'batch_mla', 'batchmla',
                              'sparse_mla', 'concat_and_cache_mla', 'cooperative_topk', 'pack_topk_indices',
                              'fwht_quant')):
        return 'mla_and_indexer'
    if any(s in key for s in ('gated_delta', 'kda', 'causal_conv')):
        return 'kda_core'
    if 'fp8' in key and 'gemm' in key:
        return 'other_fp8_gemm'
    return 'other'


def summarize(path, target_graph_only=False):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as trace:
        events = [e for e in json.load(trace)['traceEvents'] if e.get('cat') == 'kernel']
    moe = [e for e in events if kernel_family(e['name']) == 'mega_moe']
    steps, extra = divmod(len(moe), 42)
    if not steps or extra:
        raise ValueError(f'Trace must contain complete target steps with 42 MegaMoE calls, found {len(moe)}')
    if target_graph_only:
        graph_ids = {e['args'].get('graph id') for e in moe} - {None, 0}
        if not graph_ids:
            raise ValueError('Trace does not identify target graph nodes')
        events = [e for e in events if e.get('args', {}).get('graph id') in graph_ids]

    counts, durations = Counter(), Counter()
    busy = 0.
    end = float('-inf')
    for event in sorted(events, key=lambda e: e['ts']):
        start = event['ts']
        stop = start + event['dur']
        busy += max(0., stop - max(start, end))
        end = max(end, stop)
        family = kernel_family(event['name'])
        counts[family] += 1
        durations[family] += event['dur']
    return dict(trace=str(path), steps=steps, kernel_union_ms=busy / (1000 * steps),
                groups={k: dict(calls=counts[k] / steps, sum_ms=durations[k] / (1000 * steps)) for k in counts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--target-graph-only', action='store_true')
    options = parser.parse_args()
    print(json.dumps([summarize(p, options.target_graph_only) for p in options.traces], indent=2))
