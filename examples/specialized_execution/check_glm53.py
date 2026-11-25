"""Compare both runners from one real prefill.

Run with torchrun on eight H200s.
"""
import argparse
import asyncio
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateMetadata
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SSMStateCapacity
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.specialized_execution import DecodePlan


def make_inputs(ids, history, pages, slots, decoding):
    batch, query = ids.shape
    hist = torch.tensor(history, device=ids.device)
    prefill = None if decoding else SSMPrefillMetadata.from_lengths([query] * batch, history).to_device('cuda')
    return ModelInputs(
        input_ids=ids.reshape(1, -1), seq_length=torch.full_like(hist, query), history_lengths=hist,
        block_offsets=pages, is_decoding=decoding, num_ignored_history=torch.zeros_like(hist),
        max_q_seqlen=query, max_kv_seqlen=max(history) + query, sum_kv_seqlen=sum(history) + batch * query,
        state_offsets=slots,
        ssm_state_meta=SSMStateMetadata(slots[:, None], torch.where(hist > 0, slots, 0), slots[:, None]),
        ssm_prefill_meta=prefill)


@torch.inference_mode()
def main(args):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    assert dist.get_world_size() == 8
    dc = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    ctx = DistContext.build(rank, dc)
    mc = ModelConfig.from_pretrained(args.model, trust_remote_code=True, dist_config=dc, device_type='cuda')
    lengths = [args.context + i % 3 for i in range(8)]
    per_seq = (max(lengths) + max(5, args.timed_steps) + 63) // 64
    cc = CacheConfig(max_batches=8, block_size=64, num_gpu_blocks=1 + 8 * per_seq, num_cpu_blocks=0,
                     num_state_caches=9, states_shapes=mc.states_shapes, max_prefill_token_num=8192,
                     cudagraph_capture_batch_sizes=[8])
    cc.compact_state_capacity = SSMStateCapacity.from_cache_config(cc, 1)
    _finalize_sparse_mla_cache_policy([mc], cc)
    if mc.update_cache_config_func:
        mc.update_cache_config_func(cc)
    bc = BackendConfig(eager_mode=args.eager, specialized_executor='glm53_r11:R11Projection',
                       moe_backend=args.moe_backend)
    agent = BaseModelAgent(args.model, mc, cc, bc, MiscConfig(language_model_only=True), ctx, DeviceContext(),
                           trust_remote_code=True)
    DeepEPBuffer.set_explicitly_destroy()
    agent.build_model()
    assert agent.specialized_execution is not None
    agent.build_cache_plans(cc)
    agent.build_graph_runner()
    agent.build_cache_engine()
    if args.prewarm:
        agent.warmup()
    candidate = agent.patched_model
    warm_keys = set(candidate._full_graph_runners)
    if args.prewarm and not args.eager:
        assert any(isinstance(key[-1], DecodePlan) for key in warm_keys), warm_keys
        assert any(not isinstance(key[-1], DecodePlan) for key in warm_keys), warm_keys
    execution = agent.specialized_execution
    with agent.all_context(), torch.cuda.stream(agent.stream):
        standard_config = replace(bc, specialized_executor=None)
        standard = get_backend().build_graph_runner(candidate.get_model(), mc, cc, standard_config, agent.device)
        pools = ([p.tensor for p in agent.cache_engine.gpu_allocation.pools]
                 + [p.tensor for p in agent.state_cache_engine.allocation.pools]
                 + [agent.ssm_state_cursor.current_ids])
        for pool in pools:
            pool.zero_()
        # Interleaved physical pages and reversed request slots exercise both mappings.
        pages = torch.arange(1, cc.num_gpu_blocks, device='cuda', dtype=torch.int32).view(per_seq, 8).T.contiguous()
        slots = torch.arange(8, 0, -1, device='cuda')

        def forward(inputs):
            inputs, _ = asyncio.run(agent._prepare_dp_v1(inputs))
            value = agent._forward_impl(inputs)['hidden_states']
            return agent.get_logits(value), inputs.specialized_plan

        first_ids = []
        for row, length in enumerate(lengths):
            tokens = (torch.arange(length, device='cuda') * (row + 1) + rank * 8 + row) % 4096 + 128
            for begin in range(0, length, 8192):
                chunk = tokens[begin:begin + 8192]
                inputs = make_inputs(chunk[None], [begin], pages[row:row + 1], slots[row:row + 1], False)
                inputs, _ = asyncio.run(agent._prepare_dp_v1(inputs))
                assert inputs.specialized_plan == 0
                out = agent._forward_impl(inputs)['hidden_states']
            first_ids.append(agent.get_logits(out[:, -1:]).argmax(-1).reshape(()))
        first_ids = torch.stack(first_ids)
        snapshot = [p.cpu() for p in pools]

        def trajectory(runner, active):
            agent.patched_model, agent.specialized_execution = runner, active
            for pool, saved in zip(pools, snapshot):
                pool.copy_(saved)
            tokens, history = first_ids.clone(), lengths.copy()
            outputs, plans = [], []
            for step in range(5):
                # Rank zero drops a request on step two; all ranks must use stock.
                rows = list(range(7 if step == 2 and rank == 0 else 8))
                if step % 2:
                    rows.reverse()
                inputs = make_inputs(tokens[rows, None], [history[i] for i in rows], pages[rows], slots[rows], True)
                logits, plan = forward(inputs)
                outputs.append(logits.detach().cpu())
                plans.append(plan)
                tokens[rows] = logits.argmax(-1).reshape(-1)
                for row in rows:
                    history[row] += 1
            return outputs, plans

        reference, _ = trajectory(standard, None)
        expected_pools = [p.cpu() for p in pools]
        actual, plans = trajectory(candidate, execution)
        assert plans == [1, 1, 0, 1, 1], plans
        logits_equal = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, reference)]
        pools_equal = [torch.equal(a.cpu().view(torch.uint8), b.view(torch.uint8))
                       for a, b in zip(pools, expected_pools)]
        keys = list(candidate._full_graph_runners)
        if not args.eager:
            assert any(isinstance(key[-1], DecodePlan) for key in keys)
            assert any(not isinstance(key[-1], DecodePlan) for key in keys)
            if args.prewarm:
                assert set(keys) == warm_keys
        result = dict(rank=rank, plans=plans, full_logits_equal=logits_equal, pools_equal=pools_equal,
                      greedy_equal=[torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(actual, reference)],
                      max_abs=[(a.float() - b.float()).abs().max().item() for a, b in zip(actual, reference)],
                      graph_keys=[str(key) for key in keys])
        if args.profile:
            for pool, saved in zip(pools, snapshot):
                pool.copy_(saved)
            profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
                        if rank == 0 else nullcontext())
            with profiler:
                forward(make_inputs(first_ids[:, None], lengths, pages, slots, True))
                torch.cuda.synchronize()
            if rank == 0:
                result['native_projection_launches'] = sum(
                    event.device_type == torch.autograd.DeviceType.CUDA and 'glm53_kda_output_r11' in event.name
                    for event in profiler.events())

        if args.timed_steps:
            # ABBA order, warm graphs, identical initial state. Both routes vote
            # through the normal DP path. Events bracket forward, head and feedback.
            for label, runner, active in [('stock1', standard, None), ('special1', candidate, execution),
                                          ('special2', candidate, execution), ('stock2', standard, None)]:
                agent.patched_model, agent.specialized_execution = runner, active
                for pool, saved in zip(pools, snapshot):
                    pool.copy_(saved)
                current = make_inputs(first_ids[:, None], lengths, pages, slots, True)
                events = []
                torch.cuda.synchronize()
                dist.barrier(group=ctx.cpu_group)
                for _ in range(args.timed_steps):
                    current, _ = asyncio.run(agent._prepare_dp_v1(current))
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    hidden = agent._forward_impl(current)['hidden_states']
                    tokens = agent.get_logits(hidden).argmax(-1)
                    current = current.step(tokens, ssm_state_cursor=agent.ssm_state_cursor)
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize()
                result[label] = [a.elapsed_time(b) for a, b in events]
            agent.patched_model, agent.specialized_execution = candidate, execution
        results = [None] * 8
        dist.all_gather_object(results, result, group=ctx.cpu_group)
        if rank == 0:
            print(json.dumps(dict(context=lengths, eager=args.eager, prewarmed=args.prewarm,
                                  moe_backend=args.moe_backend, ranks=results)), flush=True)
        assert all(all(row['full_logits_equal']) and all(row['pools_equal']) for row in results)
        if args.profile:
            assert results[0]['native_projection_launches'] == 34, results[0]['native_projection_launches']
        # The two graph runners borrow one set of process-wide communication resources.
        del standard
        agent.release()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--context', type=int, default=63)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--prewarm', action='store_true', help='Capture both routes in normal engine warmup.')
    parser.add_argument('--profile', action='store_true', help='Count launches of the native R11 kernel on rank 0.')
    parser.add_argument('--timed-steps', type=int, default=0, help='Measure two interleaved stock/provider pairs.')
    parser.add_argument('--moe-backend', choices=['default', 'deepgemm_mega_moe', 'flashinfer_mega_moe'],
                        default='default')
    main(parser.parse_args())
