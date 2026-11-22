# Copyright (c) OpenMMLab. All rights reserved.
"""FlashKDA prefill for GLM's compact, indexed recurrent state on SM90."""

from functools import cache

import torch
import triton
import triton.language as tl

from lmdeploy.utils import get_logger

from .gated_delta_rule import _state_select


@triton.jit
def _export_states(Pool, Final, Checkpoint, Sources, Destinations, CheckpointIds, Offsets, Cu,
                   unit_stride, num_units, INNER: tl.constexpr, BLOCK: tl.constexpr,
                   HAS_CHECKPOINT: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Cu + row + 1) - tl.load(Cu + row)
    src = tl.load(Sources + row)
    dst = tl.load(Destinations + row)
    if (length > 0) & (src >= 0) & (src < num_units):
        dense = row.to(tl.int64) * INNER + lanes
        final = tl.load(Final + dense, lanes < INNER, 0)
        if (dst > 0) & (dst < num_units):
            tl.store(Pool + dst.to(tl.int64) * unit_stride + lanes, final, lanes < INNER)
        if HAS_CHECKPOINT:
            cp = tl.load(CheckpointIds + row)
            offset = tl.load(Offsets + row)
            if (cp > 0) & (cp < num_units) & (cp != src) & (cp != dst):
                # A terminal checkpoint can have a partial last tile. The
                # provider only guarantees interior snapshots on tile boundaries.
                if offset == length:
                    tl.store(Pool + cp.to(tl.int64) * unit_stride + lanes, final, lanes < INNER)
                elif (offset > 0) & (offset < length) & (offset % 64 == 0):
                    value = tl.load(Checkpoint + dense, lanes < INNER, 0)
                    tl.store(Pool + cp.to(tl.int64) * unit_stride + lanes, value, lanes < INNER)


@cache
def load_flashkda_prefill(library: str):
    """Load only the stable-ABI operators from the selected FlashKDA build."""
    torch.ops.load_library(library)
    get_size = torch.ops._flashkda_C.get_workspace_size.default
    forward = torch.ops._flashkda_C.fwd.default
    get_logger('lmdeploy').info('Loaded SM90 FlashKDA prefill operators from %s.', library)

    def prefill(q, k, v, gate, beta, pool, sources, destinations, meta, a_log, dt_bias, lower_bound):
        _, tokens, heads, dim = q.shape
        out = torch.empty_like(gate)
        if tokens == 0:
            return out
        rows = sources.numel()
        # Snapshot every input before exporting any writable or aliased unit.
        initial = _state_select(pool.unsqueeze(1), sources, torch.zeros_like(sources))
        final = torch.empty_like(initial)
        checkpoint = torch.empty_like(initial) if meta.checkpoint_ids is not None else None
        workspace = q.new_empty(get_size(tokens, heads, rows), dtype=torch.uint8)
        forward(q, k, v, gate, beta, dim**-0.5, out, workspace, a_log.view(heads), dt_bias.view(heads, dim),
                lower_bound, initial, final, meta.cu_seqlens, checkpoint, meta.checkpoint_offsets)
        inner = heads * dim * dim
        _export_states[(rows, triton.cdiv(inner, 1024))](
            pool, final, checkpoint, sources, destinations.contiguous(), meta.checkpoint_ids,
            meta.checkpoint_offsets, meta.cu_seqlens, pool.stride(0), pool.size(0),
            INNER=inner, BLOCK=1024, HAS_CHECKPOINT=checkpoint is not None)
        return out

    return prefill
