"""Reproduce the frozen R11 device-graph workload through LMDeploy plan
selection."""

import argparse
import asyncio
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from check_glm53 import make_inputs
from glm53_r11_artifact import whole_step

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SSMStateCapacity
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def prompt_ids(tokenizer, rank, row, length=65536):
    text = (
        f'这是编号 {rank * 8 + row} 的系统性能研究记录。请阅读以下资料，最后解释推理系统的工作过程。\n'
        'GPU 推理包括矩阵计算、注意力、专家路由、通信和缓存管理。每个请求保持独立状态。'
        f'实验组 {rank} 样本 {row}，下面是研究笔记。\n'
    )
    words = tokenizer.encode(text, add_special_tokens=False)
    tail = tokenizer.encode('\n请结合这些资料，详细说明如何分析计算、通信和内存性能。\n', add_special_tokens=False)
    return (words * (length // len(words) + 1))[: length - len(tail)] + tail


class FixedBatch:
    """Agree once for a fixed cohort before entering the historical device
    loop."""

    def __init__(self, agent, expected):
        self.agent = agent
        self.expected = expected
        self.bound = None
        self.get_logits = agent.get_logits
        self.ssm_state_cursor = agent.ssm_state_cursor

    def _forward_impl(self, inputs):
        if inputs is not self.bound:
            assert not torch.cuda.is_current_stream_capturing()
            self.bound, _ = asyncio.run(self.agent._prepare_dp_v1(inputs))
            assert self.bound is inputs and (inputs.specialized_plan or 0) == self.expected
        return self.agent._forward_impl(inputs)


def tensor_digest(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


@torch.inference_mode()
def main(args):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(8)
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    assert dist.get_world_size() == 8
    dc = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    ctx = DistContext.build(rank, dc)
    mc = ModelConfig.from_pretrained(args.model, trust_remote_code=True, dist_config=dc, device_type='cuda')
    width = (65536 + 1024 + 63) // 64
    cc = CacheConfig(
        max_batches=8,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1 + 8 * width,
        num_state_caches=9,
        max_prefill_token_num=8192,
        cudagraph_capture_batch_sizes=[8],
        states_shapes=mc.states_shapes,
    )
    cc.compact_state_capacity = SSMStateCapacity.from_cache_config(cc, 1)
    _finalize_sparse_mla_cache_policy([mc], cc)
    mc.update_cache_config_func(cc)
    bc = BackendConfig(eager_mode=False, specialized_executor=None if args.route == 'standard' else args.executor)
    agent = BaseModelAgent(
        args.model, mc, cc, bc, MiscConfig(language_model_only=True), ctx, DeviceContext(), trust_remote_code=True
    )
    DeepEPBuffer.set_explicitly_destroy()
    agent.build_model()
    assert args.route == 'standard' or agent.specialized_execution is not None
    agent.build_cache_plans(cc)
    agent.build_graph_runner()
    agent.build_cache_engine()
    candidate, execution = agent.patched_model, agent.specialized_execution
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['GLM53_DEVICE_LOOP_LIBRARY'] = str(Path(whole_step.__file__).with_name('libdevice_loop.so'))
    # Cache allocation initializes the reserved cursor on the construction stream.
    agent.stream.wait_stream(torch.cuda.current_stream())
    with agent.all_context(), torch.cuda.stream(agent.stream):
        standard = candidate
        if args.route == 'compare':
            standard = get_backend().build_graph_runner(
                candidate.get_model(), mc, cc, replace(bc, specialized_executor=None), agent.device
            )
        pools = (
            [p.tensor for p in agent.cache_engine.gpu_allocation.pools]
            + [p.tensor for p in agent.state_cache_engine.allocation.pools]
            + [agent.ssm_state_cursor.current_ids]
        )
        for pool in pools:
            pool.zero_()
        slots = torch.arange(1, 9, device='cuda')
        pages = torch.arange(1, cc.num_gpu_blocks, device='cuda', dtype=torch.int32).view(8, width)
        prompts = [prompt_ids(agent.tokenizer, rank, i) for i in range(8)]
        if args.frozen:
            manifest = json.loads((args.frozen / f'prompt_manifest_rank{rank}.json').read_text())
            assert manifest == dict(lengths=list(map(len, prompts)), first_tokens=[row[:32] for row in prompts])
        first = []
        for row, prompt in enumerate(prompts):
            ids = torch.tensor(prompt, device='cuda')
            for pos in range(0, 65536, 8192):
                inp = make_inputs(ids[None, pos : pos + 8192], [pos], pages[row : row + 1], slots[row : row + 1], False)
                inp, _ = asyncio.run(agent._prepare_dp_v1(inp))
                hidden = agent._forward_impl(inp)['hidden_states']
            first.append(agent.get_logits(hidden[:, -1:]).argmax(-1).view(1))
        initial = make_inputs(torch.cat(first)[:, None], [65536] * 8, pages, slots, True)
        saved_pools = [p.cpu() for p in pools]

        def restore():
            for pool, snapshot in zip(pools, saved_pools):
                pool.copy_(snapshot)
            torch.cuda.synchronize()
            dist.barrier(group=ctx.cpu_group)

        if args.prefill_snapshot:
            snapshot = torch.load(args.prefill_snapshot / f'prefill-rank{rank}.pt', weights_only=True, mmap=True)
            if len(snapshot['pools']) != len(pools) or any(
                s.shape != t.shape or s.dtype != t.dtype for s, t in zip(snapshot['pools'], pools)
            ):
                raise ValueError('Frozen and current physical cache schemas do not match')
            comparison = dict(
                current_first=initial.input_ids.flatten().cpu().tolist(),
                archived_first=snapshot['first_ids'].tolist(),
                fresh_vs_archived_pool_equal=[torch.equal(s, t) for s, t in zip(saved_pools, snapshot['pools'])],
            )
            (args.output / f'prefill-comparison-rank{rank}.json').write_text(json.dumps(comparison) + '\n')
            saved_pools = snapshot['pools']
            initial = make_inputs(snapshot['first_ids'].to(slots.device)[:, None], [65536] * 8, pages, slots, True)
        reference = None
        if args.route == 'full':
            reference = json.loads((args.reference / f'standard_rank{rank}_checks.json').read_text())
        first_run = 1 if args.route == 'full' else 0
        last_run = 0 if args.route == 'standard' else args.repeats
        cases = [
            ('standard' if index == 0 else f'r11_{index}', index, 'gpu') for index in range(first_run, last_run + 1)
        ]
        if args.route != 'standard':
            cases.append(('r11_host_replay', 1, 'host_async'))
        for name, index, mode in cases:
            agent.patched_model, agent.specialized_execution = (
                (standard, None) if index == 0 else (candidate, execution)
            )
            current = initial.clone()
            restore()
            for _ in range(32):
                current, _ = asyncio.run(agent._prepare_dp_v1(current))
                hidden = agent._forward_impl(current)['hidden_states']
                current = current.step(agent.get_logits(hidden).argmax(-1), ssm_state_cursor=agent.ssm_state_cursor)
            restore()
            _, _, tokens, logits, wall = whole_step.execute(
                FixedBatch(agent, int(index > 0)),
                initial,
                restore,
                dict(name=name, steps=512, logit_steps=512, whole_step=mode),
                args.output,
                rank,
            )
            if index and hasattr(execution.executor, 'route_counts'):
                import numpy as np

                audit = execution.executor
                stop = audit.count.item()
                indices = torch.arange(stop - 512, stop, device=slots.device) % audit.route_counts.size(0)
                np.savez_compressed(
                    args.output / f'{name}-routes-rank{rank}.npz', counts=audit.route_counts[indices].cpu().numpy()
                )
            ids = torch.stack(tokens).cpu()
            # One digest per full BF16 row-set / step retains first-mismatch localization.
            hashes = [tensor_digest(row) for row in logits]
            assert len(hashes) == 512 and logits[0].dtype == torch.bfloat16
            finite = all(torch.isfinite(t.cpu()).all().item() for t in logits)
            del logits, tokens
            result = dict(
                tokens=ids.tolist(),
                logits_sha256=hashes,
                finite=finite,
                wall_s=wall,
                cursor=agent.ssm_state_cursor.current_ids.cpu().tolist(),
                pool_sha256=[tensor_digest(p) for p in pools],
                state_sha256=[tensor_digest(p) for p in agent.state_cache_engine.state_caches],
            )
            if index:
                dot = (args.output / f'{name}_rank{rank}.dot').read_text()
                expected_nodes = {
                    'glm53_kda_output_r11': 34,
                    'fused_recurrent_gated_delta_rule_transposed_main_kernel': 34,
                    'prepare_mailbox': 42,
                    'routed_r11_cu': 42,
                    'topk_cluster': 11,
                    '_score_pages': 11,
                    '_hc_post_expand_kernel': 90,
                }
                result['graph_nodes'] = {node: dot.count(node) for node in expected_nodes}
                assert result['graph_nodes'] == expected_nodes, result['graph_nodes']
            if args.frozen:
                control = json.loads((args.frozen / f'baseline_all512_rank{rank}_metrics.json').read_text())
                result['frozen_tokens_equal'] = result['tokens'] == control['tokens']
            (args.output / f'{name}_rank{rank}_checks.json').write_text(json.dumps(result) + '\n')
            if reference is None:
                reference = result
            assert finite and all(
                result[k] == reference[k] for k in ('tokens', 'logits_sha256', 'pool_sha256', 'state_sha256')
            )
            if args.require_frozen_tokens:
                assert result['frozen_tokens_equal']
            if rank == 0:
                print(name, wall * 1000 / 512, flush=True)
        if args.kernel_trace:
            assert args.route != 'standard'
            restore()
            current = initial.clone()
            for _ in range(48):
                current, _ = asyncio.run(agent._prepare_dp_v1(current))
                hidden = agent._forward_impl(current)['hidden_states']
                current = current.step(agent.get_logits(hidden).argmax(-1), ssm_state_cursor=agent.ssm_state_cursor)
            saved_pools = [p.cpu() for p in pools]
            profiler = (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                )
                if rank == 0
                else nullcontext()
            )
            with profiler:
                whole_step.execute(
                    FixedBatch(agent, 1),
                    current,
                    restore,
                    dict(name='trace_fixed', steps=32, logit_steps=32, whole_step='host_async'),
                    args.output,
                    rank,
                )
            if rank == 0:
                profiler.export_chrome_trace(str(args.output / 'fixed-trace.json.gz'))
            dist.barrier(group=ctx.cpu_group)
        del standard
        agent.release()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--kernel-trace', action='store_true', help='Profile 32 separate fixed host-queued replays on rank 0'
    )
    parser.add_argument('--executor', default='glm53_full_r11:FullR11')
    parser.add_argument('--route', choices=('standard', 'full', 'compare'), default='compare')
    parser.add_argument('--reference', type=Path, help='Separate ordinary worker output for --route full')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--require-frozen-tokens', action='store_true')
    parser.add_argument('--prefill-snapshot', type=Path, help='Diagnostic replay from archived real prefill pools')
    parser.add_argument('--frozen', type=Path, help='Optional original raw_recheck directory for token verification')
    main(parser.parse_args())
