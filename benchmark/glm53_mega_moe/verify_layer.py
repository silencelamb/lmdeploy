"""Check a real GLM-5.3 FP8 MoE layer on eight H200s and time graph replays."""

import argparse
import contextlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.cuda.moe.mega_moe import DeepGemmMegaMoE, FlashInferMegaMoE, build_sm90_mega_moe
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.config import DistConfig, ModelConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.model_inputs import BuildModelContext, StepContextManager, set_step_ctx_manager
from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight, process_weights_after_loading


def load_layer(layer, path, layer_index):
    prefix = f'model.language_model.layers.{layer_index}.mlp.'
    weight_map = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    projections = {'gate_proj': ('gate_up', 'gate'), 'up_proj': ('gate_up', 'up'), 'down_proj': ('down', 'down')}
    assignments = {}
    for expert in layer.experts.expert_list:
        for source, (dest, shard_id) in projections.items():
            for suffix in ('weight', 'weight_scale_inv'):
                assignments[f'experts.{expert}.{source}.{suffix}'] = (
                    layer.get_parameter(f'experts.{dest}.{suffix}'), dict(expert_id=expert, shard_id=shard_id))
    for source in ('gate_proj', 'up_proj', 'down_proj'):
        dest = 'gate_up_proj' if source != 'down_proj' else source
        kwargs = dict(shard_id=int(source == 'up_proj')) if dest == 'gate_up_proj' else {}
        for suffix in ('weight', 'weight_scale_inv'):
            assignments[f'shared_experts.{source}.{suffix}'] = (
                layer.get_parameter(f'shared_experts.{dest}.{suffix}'), kwargs)
    for name in ('gate.weight', 'gate.e_score_correction_bias'):
        assignments[name] = layer.get_parameter(name), {}

    with contextlib.ExitStack() as stack:
        files = {}
        for name, (param, kwargs) in assignments.items():
            key = prefix + name
            shard = weight_map[key]
            if shard not in files:
                files[shard] = stack.enter_context(safe_open(path / shard, framework='pt'))
            load_weight(param, files[shard].get_tensor(key), **kwargs)
    process_weights_after_loading(layer)


def accuracy(actual, reference, rows):
    finite = bool(torch.isfinite(actual).all())
    inactive_max = float(actual[rows:].float().abs().amax()) if rows < actual.shape[0] else 0.0
    actual, reference = actual[:rows].float(), reference[:rows].float()
    if not rows:
        return dict(rows=0, finite=finite, relative_l2=0.0, max_abs=0.0, inactive_max_abs=inactive_max)
    error = actual - reference
    return dict(rows=rows, finite=finite, inactive_max_abs=inactive_max,
                relative_l2=float(error.norm() / reference.norm().clamp_min(1.e-20)),
                max_abs=float(error.abs().amax()))


def check_weight_packing(layer, baseline):
    """Undo the packed FC1 layout for one local checkpoint expert."""
    (packed, scale), (down, _) = layer.mega_moe_weights
    original = baseline.experts.gate_up.weight[0]
    original_scale = baseline.experts.gate_up.weight_scale_inv[0]
    two_i, hidden = original.shape
    recovered = packed[0].reshape(-1, 2, 8, hidden).transpose(0, 1).reshape(two_i, hidden)
    recovered_scale = scale[0].reshape(-1, 2, hidden // 128).transpose(0, 1).reshape(two_i // 128, hidden // 128)

    def dequantize(weight, scale):
        return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)

    return dict(payload_exact=torch.equal(original.view(torch.uint8), recovered.view(torch.uint8)),
                scale_exact=torch.equal(original_scale, recovered_scale),
                fc1_aliases_checkpoint=packed.data_ptr() == layer.experts.gate_up.weight.data_ptr(),
                fc2_aliases_checkpoint=down.data_ptr() == layer.experts.down.weight.data_ptr(),
                dequantized=accuracy(dequantize(recovered, recovered_scale),
                                     dequantize(original, original_scale), two_i))


def graph_time(fn, iters):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value = fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return value, times


def eager_time(fn, iters):
    fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(3):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iters)
    return timings


def profile_layer(layer, baseline, inputs, output, rank):
    graphs = {}
    outputs = []
    for name, module in [('default', baseline), ('mega_moe', layer)]:
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(module(inputs))
        graphs[name] = graph
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for name, graph in graphs.items():
            dist.barrier()
            for _ in range(4):
                with torch.profiler.record_function(name):
                    graph.replay()
            torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / f'layer-profile-rank{rank}.json.gz'))


@torch.inference_mode()
def run(args):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('gloo', timeout=timedelta(minutes=12))
    dist_config = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    dist_ctx = DistContext.build(rank, dist_config)
    model = ModelConfig.from_pretrained(str(args.model), trust_remote_code=True, dist_config=dist_config)
    config = model.hf_config.text_config
    config.quantization_config = model.hf_config.quantization_config
    capacity = args.capacity if args.capacity is not None else max(args.batches)
    if capacity < max(args.batches):
        raise ValueError('--capacity must cover the largest benchmark batch.')
    build = BuildModelContext(max_batch_size=capacity, quant_config=model.quant_config,
                              moe_backend=args.backend)
    manager = StepContextManager(build)
    set_step_ctx_manager(manager)
    args.output.mkdir(parents=True, exist_ok=True)

    with get_dist_manager().context(dist_ctx), build_model_context(build):
        layer_index = config.first_k_dense_replace if args.layer is None else args.layer
        if not 0 <= layer_index < config.num_hidden_layers or config.mlp_layer_types[layer_index] != 'sparse':
            raise ValueError('--layer must identify a routed MoE layer.')
        baseline = Glm5NextMoE(config, layer_index, dtype=torch.bfloat16, device=device).eval()
        load_layer(baseline, args.model, layer_index)
        measure_shared = args.shared_ablation or envs.glm53_deepgemm_fuse_shared
        # Keep a routed-only runner for the EP oracle and add shared fusion as
        # a paired full-layer measurement when selected.
        with patch.object(envs, 'glm53_deepgemm_fuse_shared', False):
            runner = build_sm90_mega_moe(config, torch.bfloat16, device, build)
        layer = Glm5NextMoE(config, layer_index, dtype=torch.bfloat16, device=device, mega_moe=runner).eval()
        load_layer(layer, args.model, layer_index)
        weight_check = check_weight_packing(layer, baseline)
        if runner.compact_weights:
            canonical = layer.experts.gate_up.weight
            if runner.weight_layout == 'compact':
                unpacked = runner.unpack_gate_up(canonical, [1] * canonical.size(0))
            else:
                experts, two_i, hidden = canonical.shape
                unpacked = canonical.reshape(experts, -1, 2, 8, hidden).transpose(1, 2).reshape(experts, two_i, hidden)
            weight_check['prefill_payload_exact'] = torch.equal(unpacked.view(torch.uint8),
                                                               baseline.experts.gate_up.weight.view(torch.uint8))
            if not (weight_check['prefill_payload_exact'] and weight_check['fc1_aliases_checkpoint']):
                raise RuntimeError('Compact FC1 did not restore the canonical prefill weights bit for bit.')
        (args.output / f'weights-rank{rank}.json').write_text(json.dumps(weight_check, indent=2) + '\n')
        if not runner.merge_gate_up_scales and not (weight_check['payload_exact'] and weight_check['scale_exact']):
            raise RuntimeError('Lossless FC1 packing changed FP8 bytes or FP32 scale values.')
        fused_shared = None
        if measure_shared and isinstance(runner, DeepGemmMegaMoE):
            fused_shared = DeepGemmMegaMoE(runner.capacity, dist_ctx.ep_gpu_group, config, fuse_shared=True)
            shared_weights = fused_shared.prepare_shared_weights(layer.shared_experts)
        dedup = None
        if args.dedup_ablation and isinstance(runner, FlashInferMegaMoE):
            dedup = FlashInferMegaMoE(runner.capacity, dist_ctx.ep_gpu_group, config, dedup_dispatch=True,
                                     exact_scales=not runner.merge_gate_up_scales, knobs=runner.knobs)
        torch.cuda.synchronize()
        dist.barrier()

        reports = []
        real_inputs = None
        if args.activations:
            real_inputs = torch.load(args.activations / f'moe-inputs-rank{rank}.pt', map_location=device,
                                     weights_only=True)[layer_index]
        for batch in args.batches:
            if real_inputs is None:
                generator = torch.Generator(device).manual_seed(1000 + rank * 100 + batch)
                inputs = torch.randn(batch, config.hidden_size, generator=generator, device=device,
                                     dtype=torch.bfloat16)
            else:
                if batch > len(real_inputs):
                    raise ValueError(f'Activation dump has only {len(real_inputs)} rows for batch {batch}.')
                inputs = real_inputs[:batch]
            valid_count = torch.empty((), device=device, dtype=torch.int32)
            ctx = SimpleNamespace(global_is_decoding=lambda: True, moe_valid_token_count=valid_count, is_dummy=False)
            with manager.context(ctx):
                routing_weights, ids = layer.gate(inputs)
                load_per_expert = torch.bincount(ids.flatten().long().cpu(), minlength=config.n_routed_experts)
                dist.all_reduce(load_per_expert)
                local_load = load_per_expert.chunk(8)[rank]
                for scenario in ('normal', 'skewed', 'empty_senders', 'all_dummy'):
                    valid = 0 if scenario == 'all_dummy' or scenario == 'empty_senders' and rank % 2 == 0 else batch
                    valid_count.fill_(valid)
                    selected = ids if scenario == 'normal' else torch.arange(
                        config.num_experts_per_tok, device=device, dtype=ids.dtype).expand(batch, -1).contiguous()

                    def native():
                        return baseline.experts(inputs, routing_weights, selected.clone())

                    def mega():
                        return runner(inputs, routing_weights, selected, layer.mega_moe_weights)

                    ref = native()
                    out = mega()
                    eager_error = accuracy(out, ref, valid)
                    captured, times = graph_time(mega, args.iterations)
                    graph_error = accuracy(captured, ref, valid)
                    _, native_times = graph_time(native, args.iterations)
                    reports.append(dict(batch=batch, case=scenario, eager=eager_error, graph=graph_error,
                                        routed_us=times, native_routed_us=native_times))
                valid_count.fill_(batch)
                candidate_layer, times = graph_time(lambda: layer(inputs[None]), args.iterations)
                native_layer, native_times = graph_time(lambda: baseline(inputs[None]), args.iterations)
                reports.append(dict(batch=batch, layer_us=times, native_layer_us=native_times,
                                    local_active_experts=int(local_load.count_nonzero()),
                                    local_max_routes=int(local_load.max()), local_routes=int(local_load.sum()),
                                    layer_accuracy=accuracy(candidate_layer.flatten(0, 1), native_layer.flatten(0, 1),
                                                            batch)))
                if isinstance(runner, FlashInferMegaMoE):
                    runner.fused_staging = False
                    upstream, upstream_times = graph_time(lambda: layer(inputs[None]), args.iterations)
                    runner.fused_staging = True
                    reports[-1]['upstream_staging_layer_us'] = upstream_times
                    reports[-1]['staging_accuracy'] = accuracy(candidate_layer.flatten(0, 1), upstream.flatten(0, 1),
                                                               batch)
                if dedup is not None:
                    layer.mega_moe = dedup
                    deduped, dedup_times = graph_time(lambda: layer(inputs[None]), args.iterations)
                    layer.mega_moe = runner
                    reports[-1]['dedup_layer_us'] = dedup_times
                    reports[-1]['dedup_accuracy'] = accuracy(deduped.flatten(0, 1), candidate_layer.flatten(0, 1),
                                                             batch)
                if fused_shared is not None:
                    def combined():
                        topk_weights, topk_ids = layer.gate(inputs)
                        return fused_shared(inputs, topk_weights, topk_ids, layer.mega_moe_weights,
                                            shared_weights=shared_weights)

                    combined_output, shared_times = graph_time(combined, args.iterations)
                    reports[-1]['fused_shared_layer_us'] = shared_times
                    reports[-1]['fused_shared_accuracy'] = accuracy(combined_output,
                                                                   native_layer.flatten(0, 1), batch)
                if args.profile and batch == 8:
                    profile_layer(layer, baseline, inputs[None], args.output, rank)

                if batch == 8:
                    # The graph stays fixed while valid token counts change
                    # independently on the eight ranks.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = runner(inputs, routing_weights, ids, layer.mega_moe_weights)
                    for count in (batch, 0, 3, 1, batch):
                        valid = count if rank % 2 == 0 else batch
                        valid_count.fill_(valid)
                        graph.replay()
                        expected = baseline.experts(inputs, routing_weights, ids.clone())
                        reports.append(dict(dynamic_valid=valid, dynamic_accuracy=accuracy(captured, expected, valid)))
                    valid_count.fill_(batch if rank % 2 == 0 else 0)
                    expected = baseline.experts(inputs, routing_weights, ids.clone())
                    valid = batch if rank % 2 == 0 else 0
                    actual = runner(inputs[:valid], routing_weights[:valid], ids[:valid], layer.mega_moe_weights)
                    reports.append(dict(eager_rows=valid, dynamic_accuracy=accuracy(actual, expected, valid)))

        if runner.compact_weights:
            ctx = SimpleNamespace(global_is_decoding=lambda: False, moe_valid_token_count=None, is_dummy=False)
            with manager.context(ctx):
                for tokens in args.prefill_batches:
                    generator = torch.Generator(device).manual_seed(1029 + rank)
                    inputs = torch.randn(tokens, config.hidden_size, generator=generator, device=device,
                                         dtype=torch.bfloat16)
                    route_weights, route_ids = baseline.gate(inputs)
                    scenarios = ['native_routes']
                    if tokens == min(args.prefill_batches):
                        scenarios.append('one_destination')
                    for case in scenarios:
                        if case == 'one_destination':
                            route_ids = torch.arange(config.num_experts_per_tok, device=device,
                                                     dtype=route_ids.dtype).expand_as(route_ids).contiguous()

                        def prefill(module):
                            return module.experts(inputs, route_weights, route_ids)

                        expected = prefill(baseline)
                        actual = prefill(layer)
                        baseline_us = eager_time(lambda: prefill(baseline), min(20, args.iterations))
                        candidate_us = eager_time(lambda: prefill(layer), min(20, args.iterations))
                        reports.append(dict(prefill_case=case, prefill_tokens=tokens,
                                            prefill_accuracy=accuracy(actual, expected, tokens),
                                            native_prefill_us=baseline_us, candidate_prefill_us=candidate_us))

        (args.output / f'rank{rank}.json').write_text(json.dumps(reports, indent=2) + '\n')
        if rank == 0:
            for report in reports:
                print(json.dumps(report), flush=True)
        failures = sum(not r[mode]['finite'] or r[mode]['relative_l2'] > args.max_relative_l2
                       or r[mode]['inactive_max_abs'] != 0.0 for r in reports if 'case' in r
                       for mode in ('eager', 'graph'))
        for row in reports:
            for name in ('layer_accuracy', 'fused_shared_accuracy', 'staging_accuracy', 'dedup_accuracy',
                         'prefill_accuracy', 'dynamic_accuracy'):
                check = row.get(name)
                if check is not None:
                    limit = .002 if name in ('staging_accuracy', 'dedup_accuracy') else args.max_relative_l2
                    if name == 'prefill_accuracy':
                        limit = 0.0
                    failures += not check['finite'] or check['relative_l2'] > limit
                    if name == 'dynamic_accuracy':
                        failures += check['inactive_max_abs'] != 0.0
        failed = torch.tensor(failures, dtype=torch.int32)
        dist.all_reduce(failed)
        if int(failed):
            raise RuntimeError(f'{int(failed)} rank/case comparisons failed the relative-L2 or finite-value check.')
        torch.cuda.synchronize()
        dist.barrier()
        for provider in (runner, fused_shared, dedup):
            if provider is not None:
                provider.workspace.destroy()
        DeepEPBuffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['deepgemm_mega_moe', 'flashinfer_mega_moe'], required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--max-relative-l2', type=float, default=0.12)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--shared-ablation', action='store_true')
    parser.add_argument('--dedup-ablation', action='store_true')
    parser.add_argument('--layer', type=int)
    parser.add_argument('--activations', type=Path)
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--prefill-batches', type=int, nargs='+', default=[128])
    run(parser.parse_args())
