# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.backends.cuda.graph_runner.standard import StandardDecoderPiecewiseGraphRuntime
from lmdeploy.pytorch.backends.cuda.moe.mega_moe import Sm90MegaMoE
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.model_inputs import DPMeta
from lmdeploy.pytorch.nn.moe.blocked_fp8 import LinearWeightsBlockedF8


def test_fixed_shape_samples_live_kv_values():
    from benchmark.glm53_mega_moe.fixed_shape import sample_kv_values
    from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
    from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views

    latent = torch.full((2, 4, 64, 1, 8), float('nan'), dtype=torch.bfloat16)
    packed = torch.zeros(2, 4, 16, 1, 8 + 4, dtype=torch.uint8)
    for layer in range(2):
        keys, scales = kpool_packed_cache_views(packed[layer], 8)
        for page in (1, 2):
            latent[layer, page, :8].fill_(layer + page)
            keys[page, :2].fill_(-2.)
            scales[page, :2].fill_(layer + page)
    offsets = torch.tensor([[2], [1]])
    actual = sample_kv_values({'k_cache': latent, DSA_INDEXER_K_CACHE_NAME: packed}, offsets, 8,
                              SimpleNamespace(index_head_dim=8, index_kpool=4))
    factors = torch.tensor([2., 1., 3., 2.])
    torch.testing.assert_close(actual['mla_latent'], factors.repeat_interleave(8 * 8))
    torch.testing.assert_close(actual['kpool_scale'], factors.repeat_interleave(2))
    torch.testing.assert_close(actual['kpool_key'], -2 * factors.repeat_interleave(2 * 8))


@pytest.mark.parametrize('backend', ['default', 'deepgemm_mega_moe', 'flashinfer_mega_moe'])
def test_sm90_mega_moe_backend_is_passed_to_the_model_builder(backend):
    cfg = PytorchEngineConfig(dp=8, ep=8, tp=1, moe_backend=backend)
    assert ConfigBuilder.build_backend_config(cfg).moe_backend == backend


@pytest.mark.parametrize('change', [dict(dp=1), dict(ep=4), dict(tp=2), dict(device_type='maca')])
def test_sm90_mega_moe_rejects_an_incompatible_topology(change):
    config = dict(dp=8, ep=8, tp=1, moe_backend='deepgemm_mega_moe') | change
    with pytest.raises(ValueError, match=r'DP8 \+ EP8 \+ TP1'):
        PytorchEngineConfig(**config)


def test_unknown_moe_backend_raises():
    with pytest.raises(ValueError, match='Unsupported PyTorch MoE backend'):
        PytorchEngineConfig(moe_backend='unknown')


def test_mega_moe_frontend_keeps_the_peer_device_order(monkeypatch):
    launcher = importlib.import_module('lmdeploy.serve.openai.launch_server')
    order = '6,5,4,3,2,1,0,7'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', order)
    monkeypatch.delenv('LMDEPLOY_EP_CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(os, 'setpgrp', lambda: None)
    seen = {}

    def mock_serve(*args, **kwargs):
        seen['frontend'] = os.environ['CUDA_VISIBLE_DEVICES']
        seen['peers'] = os.environ['LMDEPLOY_EP_CUDA_VISIBLE_DEVICES']

    monkeypatch.setattr(launcher, 'serve', mock_serve)
    config = PytorchEngineConfig(dp=8, ep=8, tp=1, moe_backend='deepgemm_mega_moe')
    launcher._run_server([2], 'glm53', backend_config=config)
    assert seen == {'frontend': '4', 'peers': order}


def test_compact_gate_up_loads_only_the_updated_expert_shard():
    gate_up = LinearWeightsBlockedF8(2, 128, 256, 'gate_up', 128, torch.float8_e4m3fn, torch.device('cpu'),
                                     expert_list=[12, 13])
    gate_up.gate_up_interleave = 8
    payload = gate_up.weight.view(torch.uint8)
    payload.zero_()
    gate = torch.arange(128 * 128, dtype=torch.int32).to(torch.uint8).view(128, 128)
    up = gate.flip(0)
    for name, values in [('gate', gate), ('up', up)]:
        gate_up.weight.weight_loader(gate_up.weight, values.view(torch.float8_e4m3fn), 13, name)
    gate_up.update_weight(gate_up.weight, gate_up.weight_scale_inv)
    packed = payload[1].view(-1, 2, 8, 128)
    assert torch.equal(packed[:, 0].reshape(128, 128), gate)
    assert torch.equal(packed[:, 1].reshape(128, 128), up)
    replacement = gate ^ 0x1F
    gate_up.weight.weight_loader(gate_up.weight, replacement.view(torch.float8_e4m3fn), 13, 'gate')
    assert torch.equal(packed[:, 0].reshape(128, 128), replacement)
    assert torch.equal(packed[:, 1].reshape(128, 128), up)
    assert not payload[0].any()


@pytest.mark.parametrize(('decoding', 'counts', 'expected'), [
    (True, (8,) * 8, 32),
    (False, (1,) * 7 + (32,), 32),
    (False, (8,) * 7 + (63,), 64),
    (False, (1,) * 7 + (128,), 256),
    (False, (8,) * 7 + (4096,), 8192),
    (False, (8,) * 7 + (8193,), None),
    (False, None, None),
])
def test_mega_moe_selects_a_common_workspace_from_dp_token_counts(decoding, counts, expected):
    runner = object.__new__(Sm90MegaMoE)
    runner.capacity = 32
    runner.prefill_capacity = 8192
    runner.prefill_runners = tuple(SimpleNamespace(capacity=cap) for cap in (64, 256, 1024, 8192))
    context = SimpleNamespace(global_is_decoding=lambda: decoding, dp_meta=DPMeta(dp_token_counts=counts))
    selected = runner.for_step(context)
    assert (selected.capacity if selected else None) == expected


def test_mixed_step_can_use_small_exact_piecewise_shapes():
    model = SimpleNamespace(piecewise_cuda_graph_token_padding=False)
    runtime = StandardDecoderPiecewiseGraphRuntime(model, 32, exact_capture_tokens=(1, 2, 4, 8, 16, 64))
    context = SimpleNamespace(enable_microbatch=False, local_adapter_ids=None)
    assert runtime.get_capture_token_sizes() == [1, 2, 4, 8, 16, 32]
    descriptor = runtime.get_piecewise_graph_descriptor(context, {'input_ids': torch.empty(1, 8)})
    assert descriptor.token_bucket == 8
    assert runtime.get_piecewise_graph_descriptor(context, {'input_ids': torch.empty(1, 9)}) is None
