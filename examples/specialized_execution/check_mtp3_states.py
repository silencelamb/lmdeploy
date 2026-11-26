"""Check native MTP3 target verify, rejection, compact state commit and
ordinary continuation on 8 H200s."""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateMetadata
from lmdeploy.pytorch.config import (
    BackendConfig,
    CacheConfig,
    DistConfig,
    MiscConfig,
    ModelConfig,
    SpecDecodeConfig,
    SSMStateCapacity,
)
from lmdeploy.pytorch.devices import DeviceContext
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.executor import _finalize_sparse_mla_cache_policy
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.spec_decode.reject_sampler import RejectionSampler
from lmdeploy.pytorch.specialized_execution import DecodePlan


def inputs(ids, history, pages, slots, meta, decode):
    n, q = ids.shape
    hist = torch.tensor(history, device=ids.device)
    return ModelInputs(
        input_ids=ids.reshape(1, -1),
        seq_length=torch.full_like(hist, q),
        history_lengths=hist,
        block_offsets=pages,
        is_decoding=decode,
        num_ignored_history=torch.zeros_like(hist),
        max_q_seqlen=q,
        max_kv_seqlen=max(history) + q,
        sum_kv_seqlen=sum(history) + q * n,
        state_offsets=slots,
        ssm_state_meta=meta,
        ssm_prefill_meta=None if decode else SSMPrefillMetadata.from_lengths([q] * n, history).to_device('cuda'),
    )


def digest(t):
    return hashlib.sha256(memoryview(t.detach().contiguous().view(torch.uint8).cpu().numpy())).hexdigest()


@torch.inference_mode()
def run(args):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('gloo', timeout=timedelta(minutes=20))
    assert dist.get_world_size() == 8
    dc = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    ctx = DistContext.build(rank, dc)
    mc = ModelConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        dist_config=dc,
        device_type='cuda',
        spec_method='deepseek_mtp',
        num_spec_tokens=3,
    )
    cc = CacheConfig(
        max_batches=2,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=16,
        num_state_caches=3,
        states_shapes=mc.states_shapes,
        max_prefill_token_num=512,
        cudagraph_capture_batch_sizes=[1, 2],
    )
    cc.compact_state_capacity = SSMStateCapacity.from_cache_config(cc, 4)
    spec = SpecDecodeConfig.from_config(
        'deepseek_mtp', 3, args.model, cc, target_model=args.model, trust_remote_code=True, dist_config=dc
    )
    _finalize_sparse_mla_cache_policy([mc], cc)
    _finalize_sparse_mla_cache_policy([spec.model_config], spec.cache_config)
    mc.update_cache_config_func(cc)
    spec.model_config.update_cache_config_func(spec.cache_config)
    options = {}
    if args.moe_backend != 'default':
        options['moe_backend'] = args.moe_backend
    bc = BackendConfig(eager_mode=args.eager, specialized_executor='glm53_r11:MTP3Projection', **options)
    agent = BaseModelAgent(
        args.model,
        mc,
        cc,
        bc,
        MiscConfig(language_model_only=True),
        ctx,
        DeviceContext(),
        trust_remote_code=True,
        specdecode_config=spec,
    )
    DeepEPBuffer.set_explicitly_destroy()
    agent.build_model()
    assert agent.specialized_execution is not None
    agent.build_cache_plans(cc, spec.cache_config)
    agent.build_graph_runner()
    agent.build_cache_engine()
    active, enabled = agent.patched_model, agent.specialized_execution
    assert agent.spec_agent.proposer.model.specialized_execution is None
    agent.stream.wait_stream(torch.cuda.current_stream())
    args.output.mkdir(exist_ok=True, parents=True)
    with agent.all_context(), torch.cuda.stream(agent.stream):
        ordinary = get_backend().build_graph_runner(
            active.get_model(), mc, cc, replace(bc, specialized_executor=None), agent.device
        )
        pools = (
            [p.tensor for p in agent.cache_engine.gpu_allocation.pools]
            + [p.tensor for p in agent.state_cache_engine.allocation.pools]
            + [agent.ssm_state_cursor.current_ids]
        )
        for p in pools:
            p.zero_()
        bank = torch.arange(1, 9, device='cuda').view(2, 4)
        slots = torch.tensor([1, 2], device='cuda')
        # Logical token pages are interleaved in storage and both rows cross page 64.
        pages = torch.tensor([[1, 3, 5], [2, 4, 6]], device='cuda', dtype=torch.int32)
        lengths = [63, 65]

        def forward(req):
            req, _ = asyncio.run(agent._prepare_dp_v1(req))
            out = agent._forward_impl(req)
            hidden = out['hidden_states']
            logits = agent.get_logits(hidden).reshape(req.seq_length.numel(), req.max_q_seqlen, -1)
            return hidden, logits, req.specialized_plan

        first = []
        for i, length in enumerate(lengths):
            m = SSMStateMetadata(bank[i : i + 1], torch.zeros_like(slots[i : i + 1]), bank[i : i + 1, :1].contiguous())
            ids = (128 + (rank * 23 + i * 7 + torch.arange(length, device='cuda')) % 2000)[None]
            _, logits, choice = forward(inputs(ids, [0], pages[i : i + 1], slots[i : i + 1], m, False))
            assert choice == 0
            first.append(logits[0, -1].argmax())
        first = torch.stack(first)
        seed_meta = SSMStateMetadata(bank, torch.zeros_like(slots), bank[:, :1].contiguous())
        meta = agent.ssm_state_cursor.advance(seed_meta, slots, torch.ones_like(slots), width=4)
        snapshot = [p.cpu() for p in pools]

        def restore():
            for p, cpu in zip(pools, snapshot):
                p.copy_(cpu)
            torch.cuda.synchronize()
            dist.barrier(group=ctx.cpu_group)

        reject = RejectionSampler('cuda')
        report = []
        for prefix in ([0, 3], [1, 2]):
            agent.patched_model, agent.specialized_execution = ordinary, None
            seq = first.new_zeros((2, 4))
            seq[:, 0] = first
            # Choose genuine matches or mismatches at each causal position. The
            # evaluated logits, acceptance and state writes are never overwritten.
            for j in range(3):
                restore()
                _, logits, _ = forward(inputs(seq, lengths, pages, slots, meta, True))
                best = logits[:, j].argmax(-1)
                change = torch.tensor([j == k for k in prefix], device='cuda')
                seq[:, j + 1] = (best + change.long()) % logits.size(-1)
            records = []
            for runner, execution in [(ordinary, None), (active, enabled)]:
                restore()
                agent.patched_model, agent.specialized_execution = runner, execution
                req = inputs(seq, lengths, pages, slots, meta, True)
                hidden, logits, choice = forward(req)
                assert (choice or 0) == int(execution is not None)
                assert torch.isfinite(logits).all()
                emitted, rejected, last = reject(
                    logits, seq[:, 1:].contiguous(), SimpleNamespace(max_top_k=1, batch_size=8)
                )
                consumed = 4 - rejected
                assert consumed.tolist() == [k + 1 for k in prefix]
                next_ids = torch.cat((last[:, None], torch.zeros_like(seq[:, 1:])), dim=1)
                follow = req.step(next_ids.reshape(1, -1), consumed, ssm_state_cursor=agent.ssm_state_cursor)
                expected = meta.output_state_ids.gather(1, (consumed - 1)[:, None]).flatten()
                assert torch.equal(follow.ssm_state_meta.input_state_ids, expected)
                assert torch.equal(agent.ssm_state_cursor.current_ids[slots], expected)
                assert follow.history_lengths.tolist() == [length + k + 1 for length, k in zip(lengths, prefix)]
                before = dict(
                    hidden=digest(hidden),
                    logits=digest(logits),
                    emitted=emitted.tolist(),
                    rejected=rejected.tolist(),
                    cursor=expected.tolist(),
                    pools=[digest(p) for p in pools],
                )
                keep = 1 if rank == 0 else 2
                m = follow.ssm_state_meta
                nxt = SSMStateMetadata(m.state_ids[:keep], m.input_state_ids[:keep], m.output_state_ids[:keep])
                continuation = inputs(
                    next_ids[:keep], follow.history_lengths[:keep].tolist(), pages[:keep], slots[:keep], nxt, True
                )
                fh, fl, choice = forward(continuation)
                assert (choice or 0) == 0
                records.append(
                    dict(
                        verify=before,
                        continuation_hidden=digest(fh),
                        continuation_logits=digest(fl),
                        final_pools=[digest(p) for p in pools],
                    )
                )
            assert records[0] == records[1], (rank, prefix, records)
            report.append(dict(accepted=prefix, equal=True, oracle=records[0]))
        trace = None
        moe_kernels = {}
        if args.profile:
            restore()
            req = inputs(seq, lengths, pages, slots, meta, True)
            from contextlib import nullcontext

            profiler = (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                )
                if rank == 0
                else nullcontext()
            )
            with profiler:
                _, _, choice = forward(req)
                assert choice == 1
                torch.cuda.synchronize()
            if rank == 0:
                trace = sum(
                    e.device_type == torch.autograd.DeviceType.CUDA and 'glm53_kda_output_r11' in e.name
                    for e in profiler.events()
                )
                assert trace == 34, trace
                from collections import Counter

                moe_kernels = dict(
                    Counter(
                        e.name
                        for e in profiler.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                        and ('mega_moe' in e.name.lower() or 'megamoe' in e.name.lower())
                    )
                )
                if args.moe_backend != 'default':
                    main_launches = sum(
                        count for name, count in moe_kernels.items() if 'deep_gemm::sm90_fp8_fp4_mega_moe_impl' in name
                    )
                    assert main_launches == 42, moe_kernels
        keys = list(active._full_graph_runners)
        if not args.eager:
            assert any(isinstance(k[-1], DecodePlan) and k[-1].num_spec_tokens == 3 for k in keys)
        (args.output / f'rank{rank}.json').write_text(
            json.dumps(
                dict(
                    cases=report,
                    mode='eager' if args.eager else 'graph',
                    native_projection_launches=trace,
                    moe_backend=args.moe_backend,
                    native_moe_kernels=moe_kernels,
                    keys=list(map(str, keys)),
                ),
                indent=2,
            )
            + '\n'
        )
        del ordinary
        agent.release()
    dist.destroy_process_group()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model')
    ap.add_argument('output', type=Path)
    ap.add_argument('--moe-backend', choices=('default', 'deepgemm_mega_moe'), default='default')
    ap.add_argument('--eager', action='store_true')
    ap.add_argument('--profile', action='store_true')
    run(ap.parse_args())
