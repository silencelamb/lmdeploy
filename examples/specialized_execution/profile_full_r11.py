"""Benchmark-only GPU timestamps and bounded CUPTI capture inside real Ray
workers."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from glm53_full_r11 import FullR11


@triton.jit
def _r11_body_start(Ticks, Count, CAP: tl.constexpr):
    i = tl.load(Count)
    now = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [i], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(Ticks + (i % CAP) * 2, now)


@triton.jit
def _r11_body_end(Ticks, Count, CAP: tl.constexpr):
    i = tl.load(Count)
    now = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [i], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(Ticks + (i % CAP) * 2 + 1, now)
    tl.store(Count, i + 1)


class ServingTrace:
    """Markers replay with the actual specialized graph, on its caller's
    stream."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.rank = ctx.dist.rank
        directory = os.getenv('GLM53_SERVING_TRACE')
        self.output = Path(directory) if directory else None
        self.count = torch.zeros(1, device=ctx.device, dtype=torch.int32)
        self.ticks = torch.zeros((8192, 2), device=ctx.device, dtype=torch.int64)
        self.monitor = None
        self.synthetic_steps = 0

    async def watch(self):
        while True:
            await asyncio.sleep(0.15)
            for tag in ('clean', 'profiled'):
                dest = self.output / f'{tag}-rank{self.rank}.json'
                if (self.output / f'dump-{tag}').is_file() and not dest.exists():
                    n = self.count.item()
                    assert n <= self.ticks.size(0)
                    values = self.ticks[:n].cpu().tolist()
                    (self.output / f'warmup-rank{self.rank}.json').write_text(json.dumps(self.synthetic_steps))
                    if hasattr(self, 'route_counts'):
                        np.savez_compressed(
                            self.output / f'{tag}-routes-rank{self.rank}.npz',
                            counts=self.route_counts[:n].cpu().numpy(),
                        )
                    tmp = dest.with_suffix('.tmp')
                    tmp.write_text(json.dumps(values))
                    tmp.replace(dest)
            if (
                self.rank == 0
                and (self.output / 'profile-at').is_file()
                and not (self.output / 'worker-trace.json.gz').exists()
            ):
                trigger = int((self.output / 'profile-at').read_text())
                if self.count.item() >= trigger:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    ) as prof:
                        await asyncio.sleep(0.35)
                    tmp = self.output / 'incomplete-trace.json.gz'
                    prof.export_chrome_trace(str(tmp))
                    tmp.replace(self.output / 'worker-trace.json.gz')

    def run(self, plan, **kwargs):
        if self.output is not None and self.monitor is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self.monitor = loop.create_task(self.watch())
        if self.model.ctx_mgr.current_context().is_dummy and not torch.cuda.is_current_stream_capturing():
            self.synthetic_steps += 1
        _r11_body_start[(1,)](self.ticks, self.count, self.ticks.size(0), num_warps=1)
        out = super().run(plan, **kwargs)
        _r11_body_end[(1,)](self.ticks, self.count, self.ticks.size(0), num_warps=1)
        return out

    def close(self):
        if self.monitor is not None:
            self.monitor.cancel()
        super().close()


class ProfiledR11(ServingTrace, FullR11):
    artifact_id = FullR11.artifact_id + '-two-timestamps'


@triton.jit
def _r11_route_counts(Plans, Counts, Step, CAP: tl.constexpr):
    layer = tl.program_id(0)
    step = (tl.load(Step) - 1) % CAP
    i = tl.arange(0, 64)
    count = tl.load(Plans + layer * 576 + 1 + i, i < 54, 0)
    tl.store(Counts + (step * 42 + layer) * 54 + i, count.to(tl.int16), i < 54)


class RouteAuditR11(ProfiledR11):
    """Record physical expert counts after the planner for a separate
    diagnostic run."""

    artifact_id = FullR11.artifact_id + '-physical-route-audit'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.route_counts = torch.empty((2048, 42, 54), dtype=torch.int16, device=ctx.device)
        self.plans = torch.empty((42, 576), device=ctx.device, dtype=torch.int32)
        self.layer_plans = {id(p): row for p, row in zip(self.moes, self.plans)}

    def moe(self, p, x, sink):
        self.ep.plan = self.layer_plans[id(p)]
        return super().moe(p, x, sink)

    def run(self, plan, **kwargs):
        out = super().run(plan, **kwargs)
        _r11_route_counts[(42,)](self.plans, self.route_counts, self.count, self.route_counts.size(0), num_warps=1)
        return out
