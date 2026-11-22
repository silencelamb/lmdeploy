"""Time native and MegaMoE prefill with real checkpoint weights and EP8."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist
from verify_layer import accuracy, eager_time, load_layer

from lmdeploy.pytorch.backends.cuda.moe.mega_moe import build_sm90_mega_moe
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
from lmdeploy.pytorch.config import DistConfig, ModelConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.model_inputs import BuildModelContext, DPMeta, StepContextManager, set_step_ctx_manager
from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE
from lmdeploy.pytorch.models.patch import build_model_context


@torch.inference_mode()
def run(args):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    dc = DistConfig(dp=8, ep=8, tp=1, dp_rank=rank)
    dist_ctx = DistContext.build(rank, dc)
    model = ModelConfig.from_pretrained(str(args.model), trust_remote_code=True, dist_config=dc)
    config = model.hf_config.text_config
    config.quantization_config = model.hf_config.quantization_config
    build = BuildModelContext(max_batch_size=32, quant_config=model.quant_config, moe_backend=args.backend)
    manager = StepContextManager(build)
    set_step_ctx_manager(manager)
    args.output.mkdir(parents=True, exist_ok=True)

    reports = []
    with get_dist_manager().context(dist_ctx), build_model_context(build):
        layer_index = args.layer if args.layer is not None else config.first_k_dense_replace
        runner = build_sm90_mega_moe(config, torch.bfloat16, device, build)
        if not runner.prefill_capacity:
            raise ValueError('Set LMDEPLOY_GLM53_MEGA_PREFILL_CAPACITY to cover the benchmark batches.')
        reference = Glm5NextMoE(config, layer_index, dtype=torch.bfloat16, device=device).eval()
        candidate = Glm5NextMoE(config, layer_index, dtype=torch.bfloat16, device=device, mega_moe=runner).eval()
        load_layer(reference, args.model, layer_index)
        load_layer(candidate, args.model, layer_index)
        generator = torch.Generator(device).manual_seed(9301 + rank)
        cases = [(str(n), (n,) * 8) for n in args.symmetric]
        cases.extend((f'mixed-{n}', (n,) + (8,) * 6 + (1,)) for n in args.mixed)
        for name, counts in cases:
            rows = counts[rank]
            dummy = name.startswith('mixed-') and rank == 7
            values = torch.randn(rows, config.hidden_size, generator=generator, device=device, dtype=torch.bfloat16)
            ctx = SimpleNamespace(global_is_decoding=lambda: False, dp_meta=DPMeta(dp_token_counts=counts),
                                  moe_valid_token_count=None, is_dummy=dummy)
            with manager.context(ctx):
                selected = runner.for_step(ctx)
                if selected is None:
                    raise ValueError(f'No MegaMoE workspace for {name}.')
                expected = reference(values[None])
                actual = candidate(values[None])
                stats = accuracy(actual.flatten(0, 1), expected.flatten(0, 1), rows if not dummy else 0)
                # Pair the three passes on each process and keep all eight ranks.
                ref_us = eager_time(lambda: reference(values[None]), args.iterations)
                new_us = eager_time(lambda: candidate(values[None]), args.iterations)
            reports.append(dict(case=name, token_counts=counts, selected_capacity=selected.capacity,
                                reference_us=ref_us, candidate_us=new_us, accuracy=stats))
            if rank == 0:
                print(json.dumps(dict(case=name, reference_us=median(ref_us), candidate_us=median(new_us),
                                      local_error=stats['relative_l2'])), flush=True)
        (args.output / f'rank{rank}.json').write_text(json.dumps(reports, indent=2) + '\n')
        failures = sum(not r['accuracy']['finite'] or r['accuracy']['relative_l2'] > args.max_relative_l2
                       for r in reports)
        failure_count = torch.tensor(failures)
        dist.all_reduce(failure_count)
        if failure_count.item():
            raise RuntimeError(f'{failure_count.item()} prefill comparisons failed.')
        torch.cuda.synchronize()
        dist.barrier()
        for prefill in runner.prefill_runners:
            prefill.workspace.destroy()
        runner.workspace.destroy()
        DeepEPBuffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', required=True, choices=('deepgemm_mega_moe', 'flashinfer_mega_moe'))
    parser.add_argument('--symmetric', nargs='+', type=int, default=[128, 512, 2048, 8192])
    parser.add_argument('--mixed', nargs='+', type=int, default=[1024, 8192])
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--max-relative-l2', type=float, default=.06)
    run(parser.parse_args())
