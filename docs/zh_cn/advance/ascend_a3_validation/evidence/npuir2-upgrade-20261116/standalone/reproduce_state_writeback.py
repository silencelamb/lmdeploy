# Copyright (c) OpenMMLab. All rights reserved.
"""Model-free recurrence/store compiler reproducer; no LMDeploy imports."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def history_kernel(initial, inputs, query, history, output, T: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0)
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    offsets = block * 4096 + rows[:, None] * 128 + cols[None, :]
    h = tl.load(initial + offsets)
    for token in range(T):
        x = tl.load(inputs + (token * BLOCKS + block) * 32 + rows)
        q = tl.load(query + token * 128 + cols)
        h = h * 0.5 + x[:, None] * q[None, :]
        o = tl.sum(h * q[None, :], 1)
        tl.store(output + (token * BLOCKS + block) * 32 + rows, o.to(tl.bfloat16))
        tl.store(history + token * BLOCKS * 4096 + offsets, h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args()
    assert not a.output.exists(), a.output
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    T = 4
    blocks = 64
    dev = "npu:0"
    initial = torch.randn(blocks, 32, 128, device="cpu").to(dev)
    inputs = torch.randn(T, blocks, 32, device="cpu").to(dev)
    query = torch.randn(T, 128, device="cpu").to(dev)
    history = torch.empty(T, blocks, 32, 128, device=dev)
    output = torch.empty(T, blocks, 32, device=dev, dtype=torch.bfloat16)
    h = initial.cpu()
    expected = []
    refout = []
    for t in range(T):
        h = h * 0.5 + inputs[t].cpu()[:, :, None] * query[t].cpu()[None, None, :]
        expected.append(h.clone())
        refout.append((h * query[t].cpu()[None, None, :]).sum(-1))
    expected = torch.stack(expected)
    refout = torch.stack(refout)

    def call():
        return history_kernel[(blocks,)](
            initial, inputs, query, history, output, T, blocks, num_warps=1, num_stages=a.stages, enable_fp_fusion=False
        )

    call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        call()
    torch.npu.synchronize()
    results = []
    for i in range(a.iterations):
        graph.replay()
        torch.npu.synchronize()
        s = history.cpu()
        o = output.cpu()
        mask = (s - expected).abs() > 1e-5
        row = dict(
            iteration=i,
            state_bad=int(mask.sum()),
            max_abs=float((s - expected).abs().max()),
            output_close=bool(torch.allclose(o.float(), refout, atol=0.1, rtol=0.01)),
        )
        if mask.any():
            row["next_iteration_equal"] = [
                int(((s[t] - expected[t + 1]).abs()[mask[t]] < 1e-5).sum()) for t in range(T - 1)
            ]
        results.append(row)
    a.output.write_text(json.dumps(dict(options=vars(a) | {"output": str(a.output)}, results=results), indent=2) + "\n")
    print(
        "MINIMAL", a.stages, "bad", sum(r["state_bad"] > 0 for r in results), "max", max(r["max_abs"] for r in results)
    )


if __name__ == "__main__":
    main()
