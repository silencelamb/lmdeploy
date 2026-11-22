"""Real-prefill GLM-5.3 fixed-shape decode, with same-token state comparisons."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateMetadata
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SSMStateCapacity
from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE
from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views


def sample(tensor, count=16384):
    flat = tensor.reshape(-1)
    return flat[::max(1, (flat.numel() + count - 1) // count)].float().cpu()


def sample_kv_values(caches, offsets, length, config):
    """Read model values at up to 32 live positions per request and cache row."""
    def locations(count, entries, token_span):
        samples = min(count, 32)
        pos = torch.arange(samples, device=offsets.device) * (count - 1) // max(samples - 1, 1)
        pages = offsets[:, pos // entries * token_span].flatten().long()
        slots = pos.remainder(entries).repeat(offsets.size(0))
        return pages, slots

    latent = caches['k_cache']
    block_size = latent.size(2)
    pages, slots = locations(length, block_size, 1)
    samples = {}
    if latent.dtype in (torch.bfloat16, torch.float16, torch.float32):
        samples['mla_latent'] = sample(latent[:, pages, slots])

    groups = length // config.index_kpool
    if groups:
        packed = caches[DSA_INDEXER_K_CACHE_NAME]
        entries = packed.size(2)
        pages, slots = locations(groups, entries, entries * config.index_kpool // block_size)
        # Unpack just the selected pages. FP8 codes and FP32 scales share
        # storage within each page, with different element sizes and strides.
        values, scales = kpool_packed_cache_views(packed[:, pages].flatten(0, 1), config.index_head_dim)
        rows = torch.arange(values.size(0), device=offsets.device)
        slots = slots.repeat(packed.size(0))
        scales = scales[rows, slots]
        samples['kpool_key'] = sample(values.float()[rows, slots] * scales)
        samples['kpool_scale'] = sample(scales)
    return samples


def compare(actual, reference):
    diff = actual.float() - reference.float()
    return dict(finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()),
                max_abs=float(diff.abs().amax()),
                rms=float(diff.square().mean().sqrt()),
                relative_l2=float(diff.norm() / reference.float().norm().clamp_min(1.e-20)))


@torch.inference_mode()
def main(args):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(6)
    dist.init_process_group('gloo', timeout=timedelta(minutes=35))
    dc = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    ctx = DistContext.build(rank, dc)
    model = ModelConfig.from_pretrained(str(args.model), trust_remote_code=True, dist_config=dc, device_type='cuda')
    per_seq = (args.context + args.steps + max(args.warmup, args.profile_steps) + 64) // 64
    cc = CacheConfig(max_batches=args.batch, block_size=64, num_cpu_blocks=0, num_gpu_blocks=1 + args.batch * per_seq,
                     num_state_caches=args.batch + 1, max_prefill_token_num=args.chunk,
                     cudagraph_capture_batch_sizes=[args.batch], states_shapes=model.states_shapes)
    cc.compact_state_capacity = SSMStateCapacity.from_cache_config(cc, 1)
    _finalize_sparse_mla_cache_policy([model], cc)
    if model.update_cache_config_func is not None:
        model.update_cache_config_func(cc)
    bc = BackendConfig(eager_mode=args.eager, moe_backend=args.backend)
    agent = BaseModelAgent(str(args.model), model, cc, bc, MiscConfig(language_model_only=True),
                           ctx, DeviceContext(), trust_remote_code=True)
    args.output.mkdir(parents=True, exist_ok=True)
    agent.build_model()
    agent.build_cache_plans(cc)
    agent.build_cache_engine()
    agent.build_graph_runner()
    record_layers = [layer for layer in agent.patched_model.get_model().modules()
                     if isinstance(layer, Glm5NextMoE) and layer.layer_idx in args.record_moe_layers]
    if {layer.layer_idx for layer in record_layers} != set(args.record_moe_layers):
        raise ValueError('--record-moe-layers must identify routed MoE layers.')
    kv = [pool.tensor for pool in agent.cache_engine.gpu_allocation.pools]
    state_pools = [pool.tensor for pool in agent.state_cache_engine.allocation.pools]
    states = agent.state_cache_engine.state_caches
    allocations = kv + state_pools + [agent.ssm_state_cursor.current_ids]
    reference = None
    if args.reference:
        reference = torch.load(args.reference / f'rank{rank}.pt', map_location='cpu', weights_only=True)
        if reference['tokens'].shape != (args.steps, args.batch):
            raise ValueError('Baseline token shape does not match --steps and --batch.')

    with agent.all_context(), torch.cuda.stream(agent.stream):
        for tensor in allocations:
            tensor.zero_()
        offsets = torch.arange(1, 1 + args.batch * per_seq, device='cuda', dtype=torch.int32).view(args.batch, per_seq)
        slots = torch.arange(1, args.batch + 1, device='cuda', dtype=torch.int64)
        tokenizer = agent.tokenizer
        prompts = []
        tail_text = '\n请结合这些资料，详细说明如何分析计算、通信和内存性能。\n'
        tail = tokenizer.encode(tail_text, add_special_tokens=False)
        for i in range(args.batch):
            text = (f'这是编号 {rank * args.batch + i} 的系统性能研究记录。'
                    '请阅读以下资料，最后解释推理系统的工作过程。\n'
                    'GPU 推理包括矩阵计算、注意力、专家路由、通信和缓存管理。每个请求保持独立状态。'
                    f'实验组 {rank} 样本 {i}，下面是研究笔记。\n')
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompts.append((ids * (args.context // len(ids) + 1))[:args.context - len(tail)] + tail)
        prompts = torch.tensor(prompts, device='cuda')
        torch.cuda.synchronize()

        def model_inputs(ids, history, which, decoding):
            batch, q = ids.shape
            local_slots = slots[which]
            inputs = ModelInputs(
                input_ids=ids.reshape(1, -1), seq_length=torch.full((batch,), q, device='cuda', dtype=torch.long),
                history_lengths=torch.full((batch,), history, device='cuda', dtype=torch.long),
                block_offsets=offsets[which], is_decoding=decoding,
                num_ignored_history=torch.zeros(batch, device='cuda', dtype=torch.long),
                max_q_seqlen=q, max_kv_seqlen=history + q, sum_kv_seqlen=(history + q) * batch,
                state_offsets=local_slots,
                ssm_state_meta=SSMStateMetadata(local_slots[:, None].contiguous(),
                                               local_slots if history else torch.zeros_like(local_slots),
                                               local_slots[:, None].contiguous()),
                ssm_prefill_meta=None if decoding else SSMPrefillMetadata.from_lengths(
                    [q] * batch, [history] * batch).to_device('cuda'))
            inputs.build_dp_meta([batch * q] * 8)
            inputs.dp_meta.dp_is_decoding = decoding
            inputs.dp_meta.dp_batches = [batch] * 8
            return inputs

        prefill_start = time.perf_counter()
        first_tokens = []
        for seq in range(args.batch):
            slot = slice(seq, seq + 1)
            for start in range(0, args.context, args.chunk):
                ids = prompts[slot, start:start + args.chunk]
                out = agent._forward_impl(model_inputs(ids, start, slot, False))
                if start + ids.shape[1] == args.context:
                    first_tokens.append(agent.get_logits(out['hidden_states'][:, -1:]).argmax(-1).reshape(1))
            torch.cuda.synchronize()
            if rank == 0:
                prefill_event = dict(phase='prefill', sequence=seq, elapsed_s=time.perf_counter() - prefill_start)
                print(json.dumps(prefill_event), flush=True)
        prefill_seconds = time.perf_counter() - prefill_start
        first_tokens = torch.cat(first_tokens)
        initial = [tensor.cpu() for tensor in allocations]
        prefill_samples = [sample(t) for t in [*kv, *states]]
        text_config = model.hf_config.text_config
        caches = agent.cache_engine.block_caches
        prefill_kv = sample_kv_values(caches, offsets, args.context, text_config)
        prefill_ids = first_tokens.cpu()
        if reference is not None:
            first_tokens = reference['prefill_ids'].to('cuda')
        chain = None if reference is None else reference['tokens'].to('cuda')
        begin = model_inputs(first_tokens[:, None], args.context, slice(None), True)

        def restore():
            for gpu, cpu in zip(allocations, initial):
                gpu.copy_(cpu)
            torch.cuda.synchronize()
            dist.barrier()

        def decode_step(inputs, token=None):
            out = agent._forward_impl(inputs)
            logits = agent.get_logits(out['hidden_states'])
            predicted = logits.argmax(-1).reshape(-1)
            next_ids = predicted if token is None else token
            return inputs.step(next_ids, ssm_state_cursor=agent.ssm_state_cursor), predicted, logits

        if record_layers:
            moe_inputs = {}

            def record(layer, inputs):
                moe_inputs[layer.layer_idx] = inputs[0].detach().flatten(0, -2).clone()

            hooks = [layer.register_forward_pre_hook(record) for layer in record_layers]
            enable_graph = agent.patched_model.enable_graph
            agent.patched_model.enable_graph = lambda **kwargs: False
            try:
                decode_step(begin.clone())
            finally:
                agent.patched_model.enable_graph = enable_graph
                for hook in hooks:
                    hook.remove()
            if set(moe_inputs) != set(args.record_moe_layers):
                raise RuntimeError('Some requested MoE layers did not run during the eager decode step.')
            torch.save({layer: value.cpu() for layer, value in moe_inputs.items()},
                       args.output / f'moe-inputs-rank{rank}.pt')
            restore()
        current = begin.clone()
        for _ in range(args.warmup):
            current, _, _ = decode_step(current)
        torch.cuda.synchronize()
        result = {}
        repeats = []
        for repeat in range(args.repeats):
            restore()
            torch.cuda.reset_peak_memory_stats()
            current = begin.clone()
            predictions, logits_samples, times, cpu_times = [], [], [], []
            for i in range(args.steps):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start_event.record()
                current, predicted, logits = decode_step(current, None if chain is None else chain[i])
                if i < args.logit_steps or i == args.steps - 1:
                    logits_samples.append(logits.clone())
                end_event.record()
                end_event.synchronize()
                times.append(start_event.elapsed_time(end_event))
                cpu_times.append((time.perf_counter() - wall_start) * 1000)
                predictions.append(predicted)
            token_ids = torch.stack(predictions).cpu()
            repeats.append(dict(device_ms=times, wall_ms=cpu_times, mean_device_ms=sum(times) / len(times),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            if repeat == 0:
                result = dict(prefill_ids=prefill_ids, tokens=token_ids, logits=torch.stack(logits_samples).cpu(),
                              prefill_samples=prefill_samples, state_samples=[sample(t) for t in [*kv, *states]])
                result['prefill_kv'] = prefill_kv
                result['decode_kv'] = sample_kv_values(caches, offsets, args.context + args.steps, text_config)
            repeats[-1]['repeat_argmax_agreement'] = float((result['tokens'] == token_ids).float().mean())
        if args.profile_steps:
            restore()
            current = begin.clone()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for i in range(args.profile_steps):
                    with torch.profiler.record_function('glm53_decode_step'):
                        current, _, _ = decode_step(current, None if chain is None else chain[i])
                    profiler.step()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(args.output / f'profile-rank{rank}.json.gz'))

        summary = dict(rank=rank, context=args.context, batch=args.batch, backend=args.backend, eager=args.eager,
                       prefill_s=prefill_seconds, repeats=repeats, gpu_blocks=cc.num_gpu_blocks,
                       state_names=[state.name for state in model.state_cache_specs], kv_pools=len(kv),
                       kv_bytes=sum(t.nbytes for t in kv),
                       cache_bytes=sum(t.nbytes for t in allocations),
                       graph_count=len(agent.patched_model._full_graph_runners))
        if not args.eager and summary['graph_count'] == 0:
            raise RuntimeError('Fixed-shape run did not capture a decode graph.')
        numeric_samples = [result['logits'], *prefill_samples[len(kv):], *result['state_samples'][len(kv):],
                           *prefill_kv.values(), *result['decode_kv'].values()]
        if not all(t.isfinite().all() for t in numeric_samples):
            raise RuntimeError('Nonfinite sampled logits, cache values or model state.')
        if reference is not None:
            summary['accuracy'] = dict(prefill_exact=bool(torch.equal(prefill_ids, reference['prefill_ids'])),
                                       prefill_token_matches=int((prefill_ids == reference['prefill_ids']).sum()),
                                       argmax_agreement=float((result['tokens'] == reference['tokens']).float().mean()),
                                       logits=compare(result['logits'], reference['logits']),
                                       states=[compare(a, b) for a, b in zip(result['state_samples'],
                                                                            reference['state_samples'], strict=True)],
                                       prefill_states=[compare(a, b) for a, b in zip(prefill_samples,
                                                                                    reference['prefill_samples'],
                                                                                    strict=True)])
            for legacy in ('states', 'prefill_states'):
                for metric in summary['accuracy'][legacy][:len(kv)]:
                    metric['domain'] = 'storage_bytes'
            for phase in ('prefill_kv', 'decode_kv'):
                if phase in reference:
                    summary['accuracy'][phase] = {key: compare(value, reference[phase][key])
                                                  for key, value in result[phase].items()}
        torch.save(result, args.output / f'rank{rank}.pt')
        (args.output / f'rank{rank}.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(rank=rank, means_ms=[r['mean_device_ms'] for r in repeats],
                              accuracy=summary.get('accuracy'), graph_count=summary['graph_count'])), flush=True)
        if reference is not None and args.check_prefill:
            check = summary['accuracy']
            prefill_errors = [*check['prefill_states'], *check.get('prefill_kv', {}).values()]
            if not check['prefill_exact'] or any(s['max_abs'] != 0 for s in prefill_errors):
                raise RuntimeError('Prefill tokens or sampled state changed relative to the reference.')
            if not check['logits']['finite'] or any(not s['finite'] for s in check['states']):
                raise RuntimeError('Decode produced nonfinite logits or state samples.')
    dist.barrier()
    agent.release()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--backend', default='default', choices=['default', 'deepgemm_mega_moe', 'flashinfer_mega_moe'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=65536)
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--logit-steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile-steps', type=int, default=0)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--check-prefill', action='store_true')
    parser.add_argument('--record-moe-layers', type=int, nargs='+', default=[])
    main(parser.parse_args())
