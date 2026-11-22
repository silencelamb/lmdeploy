"""Compare indexed KDA prefill with identical packed BF16 inputs on one H200."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from lmdeploy.pytorch.backends.cuda.flashkda import load_flashkda_prefill
from lmdeploy.pytorch.compact_state import SSMPrefillMetadata
from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_kda


def relative_error(output, reference):
    y = reference.float()
    return ((output.float() - y).norm() / y.norm().clamp_min(1e-20)).item()


@torch.inference_mode()
def measure(flashkda, lengths, rounds, iterations):
    heads, dim, rows = 64, 128, len(lengths)
    qkv = torch.randn(1, sum(lengths), 3, heads, dim, device='cuda', dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)
    gate = torch.randn_like(q).contiguous()
    beta = torch.randn(q.shape[:3], device='cuda', dtype=q.dtype)
    a_log = torch.randn(heads, device='cuda') / 8
    bias = torch.randn(heads * dim, device='cuda')
    src = torch.arange(1, 1 + rows, device='cuda')
    dst = src + rows
    checkpoint = list(range(2 * rows + 1, 3 * rows + 1))
    offsets = [min(n, max(64, n // 128 * 64)) for n in lengths]
    meta = SSMPrefillMetadata.from_lengths(
        lengths, [64] * rows, checkpoint_ids=checkpoint, checkpoint_offsets=offsets).to_device('cuda')
    storage = torch.randn(3 * rows + 1, 2, heads, dim, dim, device='cuda') / 8
    # Strided per-layer views. All writes go to disjoint destinations, so graph
    # replay starts from the same inputs without restoring cache in the timer.
    fla_pool, flash_pool = storage[:, 0], storage.clone()[:, 0]

    def fla():
        return indexed_chunk_kda(
            q, k, v, gate, beta, fla_pool, src, dst, A_log=a_log, dt_bias=bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            safe_gate=True, lower_bound=-5., cu_seqlens=meta.cu_seqlens, chunk_indices=meta.chunk_indices,
            chunk_offsets=meta.chunk_offsets, checkpoint_ids=meta.checkpoint_ids,
            checkpoint_offsets=meta.checkpoint_offsets)

    def fused():
        return flashkda(q, k, v, gate, beta, flash_pool, src, dst, meta, a_log, bias, -5.)

    expected, actual = fla(), fused()
    errors = dict(output=relative_error(actual, expected), final=relative_error(flash_pool[dst], fla_pool[dst]),
                  checkpoint=relative_error(flash_pool[checkpoint], fla_pool[checkpoint]))
    graphs, times = {}, {'fla': [], 'flashkda': []}
    for name, forward in [('fla', fla), ('flashkda', fused)]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        graph.replay()
        graphs[name] = graph
    for i in range(rounds):
        for name in (graphs if i % 2 == 0 else reversed(graphs)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) / iterations)
    return dict(lengths=lengths, relative_l2=errors, graph_ms=times,
                median_ms={name: statistics.median(samples) for name, samples in times.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, help='Path to the FlashKDA stable-ABI shared object.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if min(args.rounds, args.iterations) < 1:
        parser.error('--rounds and --iterations must be positive')
    torch.manual_seed(42)
    flashkda = load_flashkda_prefill(args.library)
    cases = [[64], [512], [1024], [8192], [1, 4096, 64, 128]]
    result = [measure(flashkda, lengths, args.rounds, args.iterations) for lengths in cases]
    args.output.write_text(json.dumps(result, indent=2) + '\n')
