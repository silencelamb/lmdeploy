# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3 H200 executor using the compact R11 KDA output projection."""
import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from lmdeploy.pytorch.specialized_execution import BuildContext, DecodePlan, SpecializedExecutor


@triton.jit
def _pack(Weight, Packed, BLOCK: tl.constexpr):
    # The original split-K boundaries and SW128 byte order are part of the cubin ABI.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split_elements: tl.constexpr = 4096 * 2752
    full_tiles: tl.constexpr = 21 * 64 * 8192
    split = i // split_elements
    inner = i % split_elements
    tail = inner >= full_tiles
    k_tile = inner // (64 * 8192)
    n_tile = tl.where(tail, (inner - full_tiles) // 4096, inner // 8192 % 64)
    half = tl.where(tail, 0, inner // 4096 % 2)
    row = i // 64 % 64
    col = i % 64 ^ ((row % 8) * 8)
    k = split * 2752 + k_tile * 128 + half * 64 + col
    word = tl.load(Weight + (n_tile * 64 + row) * 8192 + k, k < 8192, 0)
    tl.store(Packed + i, word)


@triton.jit
def _reduce(Partials, Output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = (tl.load(Partials + i) + tl.load(Partials + N + i)) + tl.load(Partials + 2 * N + i)
    tl.store(Output + i, acc.to(tl.bfloat16))


def _check(status):
    if status:
        raise RuntimeError(f'R11 KDA projection returned CUDA driver error {status}')


class R11Projection(SpecializedExecutor):
    api_version = 1
    artifact_id = 'glm53-r11-kda-3eaf249a-host-v1'

    @classmethod
    def describe(cls, ctx: BuildContext):
        from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration

        dist = ctx.dist.dist_config
        if (not isinstance(ctx.model, Glm5NextForConditionalGeneration)
                or ctx.model_config.dtype != torch.bfloat16 or ctx.device.type != 'cuda'
                or (dist.dp, dist.ep, dist.attn_tp, dist.mlp_tp, dist.dcp) != (8, 8, 1, 1, 1)
                or torch.cuda.get_device_capability(ctx.device) != (9, 0)
                or 'H200' not in torch.cuda.get_device_name(ctx.device)):
            return ()
        layers = ctx.model.model.layers
        projections = [layer.self_attn.o_proj for layer in layers if layer.is_linear_attention]
        if (len(layers) != 45 or len(projections) != 34
                or any(p.weight.shape != (4096, 8192) or p.weight.dtype != torch.bfloat16
                       or not p.weight.is_contiguous() or p.weight.device != ctx.device
                       or p.bias is not None or p.tp != 1 or p.lora_adapters
                       for p in projections)):
            return ()
        return (DecodePlan('r11-kda-output', batch_size=8, max_kv_len=66048, graph_safe=True),)

    def __init__(self, ctx, *, native_dir=None):
        super().__init__(ctx)
        root = native_dir if native_dir is not None else Path(os.environ['GLM53_R11_PROJECTION_DIR'])
        self.driver = ctypes.CDLL(str(root / 'libglm53_projection_v1.so'))
        address = ctypes.c_void_p
        self.driver.glm53_projection_open_v1.argtypes = [ctypes.c_char_p, ctypes.POINTER(address),
                                                         ctypes.POINTER(address)]
        self.driver.glm53_output_launch.argtypes = [address] * 5
        self.driver.glm53_output_close.argtypes = [address]
        self.owner, self.function = address(), address()
        self.device = ctx.device
        self.weights = {}
        with torch.cuda.device(self.device):
            for layer in self.model.model.layers:
                if layer.is_linear_attention:
                    weight = layer.self_attn.o_proj.weight
                    packed = torch.empty_like(weight)
                    _pack[(weight.numel() // 512,)](weight.view(torch.int16), packed.view(torch.int16), 512)
                    self.weights[layer.self_attn.layer_idx] = packed
            torch.cuda.synchronize(self.device)
            _check(self.driver.glm53_projection_open_v1(os.fsencode(root / 'kda_output.cubin'),
                                                        ctypes.byref(self.owner), ctypes.byref(self.function)))

    def project(self, layer_idx, hidden):
        x = hidden.view(8, 8192)
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != self.device:
            raise ValueError('Invalid R11 projection activation')
        partials = x.new_empty((3, 8, 4096), dtype=torch.float32)
        output = x.new_empty((1, 8, 4096))
        _check(self.driver.glm53_output_launch(self.function, x.data_ptr(), self.weights[layer_idx].data_ptr(),
                                               partials.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream))
        self.reduce_partials(partials, output)
        return output

    @staticmethod
    def reduce_partials(partials, output):
        _reduce[(128,)](partials, output, 8 * 4096, 256)

    def run(self, plan, **model_inputs):
        return self.model(**model_inputs, kda_output_projection=self.project)

    def close(self):
        with torch.cuda.device(self.device):
            _check(self.driver.glm53_output_close(self.owner))
        self.weights.clear()
