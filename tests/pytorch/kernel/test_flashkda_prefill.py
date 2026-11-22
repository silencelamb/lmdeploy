# Copyright (c) OpenMMLab. All rights reserved.
"""Check FlashKDA's packed activations and compact state ownership."""
import os

import pytest
import torch


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.bfloat16])
def test_indexed_flashkda_prefill(checkpoint, state_dtype):
    library = os.getenv('LMDEPLOY_GLM53_FLASHKDA_LIBRARY')
    if not library or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('Set LMDEPLOY_GLM53_FLASHKDA_LIBRARY on SM90 to run this provider test.')
    from lmdeploy.pytorch.backends.cuda.flashkda import load_flashkda_prefill
    from lmdeploy.pytorch.compact_state import SSMPrefillMetadata
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_kda

    run = load_flashkda_prefill(library)
    torch.manual_seed(42)
    lengths = [0, 137, 79, 67, 65, 32, 81]
    heads, dim = 8, 128
    packed = torch.randn(1, sum(lengths), 3, heads, dim, device='cuda', dtype=torch.bfloat16)
    q, k, v = packed.unbind(2)
    gate = torch.randn_like(q).contiguous()
    beta = torch.randn(1, sum(lengths), heads, device='cuda', dtype=q.dtype)
    a_log = torch.randn(heads, device='cuda') / 8
    dt_bias = torch.randn(heads * dim, device='cuda')
    # Covers an empty row, shared sources, in-place advance, terminal snapshot
    # (both separate and coalesced), read-only unit zero and invalid sources.
    src = torch.tensor([1, 2, 3, 3, 0, -1, 99], device='cuda')
    dst = torch.tensor([1, 2, 4, 5, 6, 7, 8], device='cuda')
    cp = [-1, 10, 11, 5, 0, 12, 13] if checkpoint else None
    offsets = [0, 64, 79, 67, 0, 32, 64] if checkpoint else None
    meta = SSMPrefillMetadata.from_lengths(
        lengths, [0] * len(lengths), checkpoint_ids=cp, checkpoint_offsets=offsets).to_device('cuda')
    backing = torch.randn(14, 2, heads, dim, dim, device='cuda', dtype=state_dtype) / 8
    backing[0].zero_()
    before = backing.clone()
    pool = backing[:, 1]

    def candidate():
        return run(q, k, v, gate, beta, pool, src, dst, meta, a_log, dt_bias, -5.)

    def reference():
        expected = before.clone()
        out = indexed_chunk_kda(
            q, k, v, gate, beta, expected[:, 1], src, dst, A_log=a_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            safe_gate=True, lower_bound=-5., cu_seqlens=meta.cu_seqlens, chunk_indices=meta.chunk_indices,
            chunk_offsets=meta.chunk_offsets, checkpoint_ids=meta.checkpoint_ids,
            checkpoint_offsets=meta.checkpoint_offsets)
        return out, expected

    def compare(actual, expected):
        x, y = actual.float(), expected.float()
        assert (x - y).norm() <= .02 * y.norm()

    empty = SSMPrefillMetadata.from_lengths([0] * len(lengths), [0] * len(lengths)).to_device('cuda')
    result = run(q[:, :0], k[:, :0], v[:, :0], gate[:, :0], beta[:, :0], pool,
                 src, dst, empty, a_log, dt_bias, -5.)
    assert result.shape == q[:, :0].shape and torch.equal(backing, before)
    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    for source in (2, 3):
        src[1] = source
        packed.mul_(.75)
        backing.copy_(before)
        output, expected = reference()
        graph.replay()
        for x, y in zip(captured.split(lengths, dim=1), output.split(lengths, dim=1)):
            compare(x, y)
        for unit in range(len(pool)):
            if torch.equal(expected[unit, 1], before[unit, 1]):
                assert torch.equal(pool[unit], before[unit, 1])
            else:
                compare(pool[unit], expected[unit, 1])
        assert torch.equal(backing[:, 0], before[:, 0])
        if checkpoint:
            assert torch.equal(pool[11], pool[4])
