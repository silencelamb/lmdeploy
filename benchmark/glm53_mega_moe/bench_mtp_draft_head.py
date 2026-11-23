"""Compare the replicated and DP-sharded draft projections using the checkpoint's BF16 vocabulary head."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from lmdeploy.pytorch.model_inputs import DPMeta
from lmdeploy.pytorch.spec_decode.proposers.deepseek_mtp import DeepseekMTP


@torch.inference_mode()
def run(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    group = dist.group.WORLD
    rank = group.rank()
    if group.size() != 8:
        raise ValueError('Launch with torchrun --nproc-per-node=8 on one H200 node.')

    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    with safe_open(args.model / index['weight_map']['lm_head.weight'], framework='pt') as file:
        weight = file.get_tensor('lm_head.weight').to(device=device, dtype=torch.bfloat16)
    head = SimpleNamespace(weight=weight, vocab_size=weight.size(0))
    proposer = object.__new__(DeepseekMTP)
    proposer.target_model = SimpleNamespace(lm_head=head, get_logits=lambda hidden: F.linear(hidden, weight))
    generator = torch.Generator(device).manual_seed(74191 + rank)
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for batch in (1, 4, 8, 12, 16, 17, 32):
            for ragged in (False, True):
                sizes = [max(0, batch - peer) for peer in range(group.size())] if ragged else [batch] * group.size()
                rows = sizes[rank]
                hidden = torch.empty((1, rows, weight.size(1)), device=device, dtype=weight.dtype)
                meta = DPMeta(dp_is_decoding=True, dp_batches=None if ragged else sizes,
                              dp_token_counts=tuple(n if ragged else n * 4 for n in sizes))
                inputs = SimpleNamespace(dp_meta=meta)

                def forward(shard):
                    proposer._draft_head_group = group if shard else None
                    return proposer._get_mtp_logits(hidden, inputs)

                max_abs = max_relative_l2 = 0.
                matched = 0
                for _ in range(12):
                    hidden.normal_(generator=generator)
                    ref, out = forward(False), forward(True)
                    assert out.shape == ref.shape == (1, rows, head.vocab_size)
                    if rows:
                        diff = out.float() - ref.float()
                        max_abs = max(max_abs, diff.abs().max().item())
                        max_relative_l2 = max(max_relative_l2, (diff.norm() / ref.float().norm()).item())
                        matched += (out.argmax(-1) == ref.argmax(-1)).sum().item()
                        hidden.zero_()
                        idx = torch.arange(rows, device=device)
                        hidden[0, idx, (19 * rank + idx) % hidden.size(-1)] = 1
                    # This input selects a single checkpoint column. Test the rank/row/vocabulary permutation exactly.
                    torch.testing.assert_close(forward(True), forward(False), rtol=0, atol=0)
                if max_relative_l2 >= 1.e-3:
                    raise ValueError(f'Unexpected relative L2 error {max_relative_l2} on rank {rank}.')

                timings = [[], []]
                if not ragged:
                    hidden.normal_(generator=generator)
                    for shard in (False, True, True, False, True, False, False, True):
                        torch.cuda.synchronize()
                        dist.barrier()
                        start = perf_counter()
                        for _ in range(args.iterations):
                            forward(shard)
                        torch.cuda.synchronize()
                        timings[shard].append((perf_counter() - start) * 1.e6 / args.iterations)
                results.append(dict(batch=batch, ragged=ragged, rows=rows, top1=[matched, 12 * rows], max_abs=max_abs,
                                    max_relative_l2=max_relative_l2, unit_inputs_exact=True,
                                    off_us=timings[0], on_us=timings[1]))
        (args.output / f'rank{rank}.json').write_text(json.dumps(results, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=60)
    options = parser.parse_args()
    if options.iterations < 1:
        parser.error('--iterations must be positive')
    run(options)
