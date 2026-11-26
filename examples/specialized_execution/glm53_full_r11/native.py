"""Launch the frozen R11 programs on the worker's current stream."""

import ctypes as c
import os
from functools import lru_cache

import torch

P = c.c_void_p
I = c.c_int


def check(status):
    if status:
        raise RuntimeError(f'R11 native launch failed with CUDA status {status}')


def function(path, name, types):
    func = getattr(c.CDLL(str(path)), name)
    func.argtypes = types
    func.restype = I
    return func


def stream():
    return torch.cuda.current_stream().cuda_stream


@lru_cache
def load_programs(directory, device):
    # The archived QKV launcher retains a CUmodule in process-local static
    # storage. A worker owns one CUDA device. Reuse immutable code on reload.
    with torch.cuda.device(device):
        return Programs(directory)


class Programs:
    def __init__(self, root):
        qkv = root / 'libqkv_sink.so'
        check(function(qkv, 'qkv_init', [c.c_char_p])(os.fsencode(root / 'qkv_sink.cubin')))
        self.qkv_launch = function(qkv, 'qkv_launch', [P] * 5)
        self.hc_launch = function(root / 'libhc_vector.so', 'hc_pre_vector', [P] * 12 + [I, I, P])
        self.topk_launch = function(root / 'libglm53_topk_r09.so', 'glm53_topk', [P] * 3 + [I] * 6 + [P])
        self.mla_launch = function(root / 'libglm53_mla.so', 'glm53_sparse_mla', [P] * 7 + [I] * 9 + [c.c_float, P])
        self.packet = function(root / 'libpacket_sink.so', 'replica_prepare_map', [P] * 13)
        check(function(root / 'librouted.so', 'r10_init', [I])(0))
        self.routed = function(root / 'librouted.so', 'r10_launch', [I, P, I, P])

    def hc(self, x, weight, buffers, base, scale, norm, qkv, defer):
        out = x.new_empty((1, 8, 4096))
        post = x.new_empty((1, 8, 4), dtype=torch.float32)
        comb = x.new_empty((1, 8, 4, 4), dtype=torch.float32)
        operands = (weight, *buffers, x, base, scale, norm, out, post, comb)
        check(
            self.hc_launch(
                *(v.data_ptr() for v in operands), qkv.data_ptr() if qkv is not None else None, 1, int(defer), stream()
            )
        )
        return out, post, comb

    def qkv(self, hidden, weight, sink):
        out = hidden.new_empty((1, 8, 24576))
        check(self.qkv_launch(hidden.data_ptr(), weight.data_ptr(), out.data_ptr(), sink.data_ptr(), stream()))
        return out

    def topk(self, scores, lengths):
        out = scores.new_empty((8, 512), dtype=torch.int32)
        check(
            self.topk_launch(
                scores.data_ptr(),
                lengths.data_ptr(),
                out.data_ptr(),
                8,
                scores.size(1),
                scores.stride(0),
                512,
                -1,
                lengths.element_size() * 8,
                stream(),
            )
        )
        return out

    def mla(self, q, kv, indices, scale):
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        maximum = q.new_empty(q.shape[:2], dtype=torch.float32)
        lse = torch.empty_like(maximum)
        check(
            self.mla_launch(
                q.data_ptr(),
                kv.data_ptr(),
                indices.data_ptr(),
                out.data_ptr(),
                maximum.data_ptr(),
                lse.data_ptr(),
                None,
                8,
                kv.size(0),
                indices.size(-1),
                *q.stride()[:2],
                *kv.stride()[:2],
                *indices.stride()[:2],
                scale,
                stream(),
            )
        )
        return out
