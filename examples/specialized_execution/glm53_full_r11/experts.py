"""R11 balanced EP8 replicas, per-source packets and 132-CTA routed grid."""

import ctypes

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from glm53_r11_artifact.tiled_weights import convert

from .native import check, stream


def pack_owned_experts(provider, experts):
    """Pack each local expert once and reserve contiguous slots for peer
    replicas."""
    sources = [(m.weight, m.weight_scale_inv) for m in (experts.gate_up, experts.down)]
    packed = []
    for weight, scale in sources:
        weight_out = weight.new_empty((54, *weight.shape[1:]))
        shape = (54, *scale.shape[1:])
        # The native R6 ABI reads the packed-weight pointer after the scale payload.
        storage = scale.new_empty(54 * scale[0].numel() + 2)
        storage[-2:].view(torch.uint64).fill_(weight_out.data_ptr())
        packed.append((weight_out, storage[:-2].view(shape)))
    for start in range(0, 36, 6):
        chunk = slice(start, start + 6)
        pairs = [(w[chunk], sf[chunk]) for w, sf in sources]
        transformed = provider.transform_weights_for_mega_moe_sm90(*pairs)
        for (target_w, target_sf), (w, sf) in zip(packed, convert(transformed, kmajor=True)):
            target_w[chunk].copy_(w)
            target_sf[chunk].copy_(sf)
    return tuple(packed)


class ExpertParallel:
    def __init__(self, provider, programs, group, device):
        self.group = group
        self.rank = dist.get_rank(group)
        self.programs = programs
        self.work = symm.empty(2048, device=device, dtype=torch.uint8)
        self.inbox = symm.empty(2 * 8 * (9 + 22) + 1, device=device, dtype=torch.int64)
        self.work.zero_()
        self.inbox.zero_()
        torch.cuda.synchronize(device)
        self.work_handle = symm.rendezvous(self.work, group)
        self.buffer = b = provider.get_symm_buffer_for_sm90_mega_moe(
            group, 432, 8, 8, 4096, 2048, fused=True, l2_act_sf_gran_k=128
        )
        if (
            b.num_ring_tokens != 0
            or b.num_max_tokens_per_rank != 128
            or b.l1_acts.shape != (15104, 4096)
            or b.l1_acts_sf.shape != (60416, 32)
        ):
            raise ValueError('Incompatible R11 expert workspace ABI')
        self.inbox_handle = symm.rendezvous(self.inbox, group)
        pool_blocks = ((8 * 128 * 8 + 54 * 127 + 127) // 128) * 128 // 8
        self.ptrs = torch.tensor(
            [
                *self.work_handle.buffer_ptrs,
                b.buffer.data_ptr(),
                pool_blocks,
                32 + 432 * 16 + 54 * 8,
                *self.inbox_handle.buffer_ptrs,
                self.rank,
            ],
            device=device,
            dtype=torch.uint64,
        )
        self.loads = torch.empty(8, device=device, dtype=torch.int32)
        self.plan = torch.empty(64 + 8 * 64, device=device, dtype=torch.int32)
        begin = b.l2_acts_sf.data_ptr() - b.buffer.data_ptr() + b.l2_acts_sf.numel() * 4
        b.buffer[begin : begin + 8 * 128 * 4096 * 2].view(torch.uint8).fill_(255)

    def exchange(self, packed):
        # Ship the exact bytes consumed by the device grid. Each expert has
        # independent tiling, so copying a packed row commutes with replica placement.
        for h, offset in enumerate((1, 3, 5)):
            ids = torch.tensor([2 * (h + 3 * j) for j in range(6)], device=self.work.device)
            ops, sends = [], []
            for pair in packed:
                for target in pair:
                    send = target.index_select(0, ids)
                    sends.append(send)
                    recv = target[36 + h * 6 : 42 + h * 6]
                    ops.append(dist.P2POp(dist.isend, send, (self.rank + offset) % 8, self.group))
                    ops.append(dist.P2POp(dist.irecv, recv, (self.rank - offset) % 8, self.group))
            for request in dist.batch_isend_irecv(ops):
                request.wait()

    def run(self, hidden, weights, ids, packed, sink):
        b = self.buffer
        tensors = (
            hidden,
            weights,
            ids,
            b.x,
            b.x_sf,
            b.topk_weights,
            self.work,
            self.ptrs,
            b.topk_idx,
            self.loads,
            self.plan,
            sink,
        )
        check(self.programs.packet(*(t.data_ptr() for t in tensors), stream()))
        (_, up_scale), (_, down_scale) = packed
        out = torch.empty_like(hidden)
        args = (
            *b.handle.buffer_ptrs,
            out.data_ptr(),
            up_scale.data_ptr(),
            down_scale.data_ptr(),
            b.l1_acts.data_ptr(),
            b.l1_acts_sf.data_ptr(),
            b.l2_acts.data_ptr(),
            b.l2_acts_sf.data_ptr(),
            self.plan.data_ptr(),
            0,
        )
        frame = (ctypes.c_uint64 * len(args))(*args)
        check(self.programs.routed(0, frame, self.rank, stream()))
        return out
