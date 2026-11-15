# perf_worker.py

```python
"""Validation-only Ray worker: switch exact old/new norm on shared weights."""
import importlib.util
import os
import sys
import time

import torch

from lmdeploy.pytorch.engine.executor.ray_executor import RayWorkerWrapper


class PerfWorker(RayWorkerWrapper):
    def norm_perf_setup(self):
        from lmdeploy.pytorch.backends.ascend.graph_runner import AscendGraphRunner
        from lmdeploy.pytorch.nn import RMSNormGated

        path = '/root/workspace/a3-gated-norm-20261115/lmdeploy-before/lmdeploy/pytorch/kernels/ascend/triton_ops/rms_norm_gated.py'
        spec = importlib.util.spec_from_file_location('perf_frozen_norm', path)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        runner = self.model_agent.patched_model
        assert isinstance(runner, AscendGraphRunner), type(runner)
        self._norm_runner = runner
        self._norm_modes = {}
        self._norm_layers = []
        with self.model_agent.all_context(), torch.inference_mode():
            for name, new in list(runner.model.named_modules()):
                if isinstance(new, RMSNormGated):
                    assert new.activation in ('silu', 'swish'), new.activation
                    assert new.weight.dtype == torch.bfloat16
                    legacy = old.RMSNormGated(new.hidden_size, eps=new.eps,
                                             norm_before_gate=True, device=new.weight.device)
                    legacy.weight = new.weight
                    legacy.bias = new.bias
                    legacy.eval()
                    assert legacy.weight is new.weight
                    self._norm_layers.append((name, legacy, new))
            assert len(self._norm_layers) == 30, len(self._norm_layers)
            self._norm_modes = {
                'new': (runner._runner_map, runner.graph_pool_handle),
                'old': ({}, torch.npu.graph_pool_handle()),
            }
            self._norm_mode = 'new'
            runner._perf_enabled = False
            runner._perf_spans = []
            # Only this validation instance gets the measurement subclass.
            class TimedRunner(AscendGraphRunner):
                def __call__(inner, **kwargs):
                    if not inner._perf_enabled:
                        return super().__call__(**kwargs)
                    context = inner.ctx_mgr.current_context()
                    decode = context.global_is_decoding()
                    graph = decode and inner.enable_graph(**kwargs)
                    key = inner.get_graph_key(**kwargs) + inner.model.get_cudagraph_extra_key(**kwargs) if graph else None
                    capture = graph and key not in inner._runner_map
                    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    began = time.perf_counter()
                    start.record()
                    result = super().__call__(**kwargs)
                    end.record()
                    inner._perf_spans.append((start, end, {
                        'host_ms': (time.perf_counter()-began)*1000,
                        'decode': decode, 'graph': graph, 'capture': capture,
                        'graph_key': key, 'tokens': kwargs['input_ids'].numel(),
                    }))
                    return result
            runner.__class__ = TimedRunner
        return {'rank': self.rank, 'pid': os.getpid(), 'norm_names': [x[0] for x in self._norm_layers],
                'shared_weights': True, 'dlinfer_modules': [x for x in sys.modules if x.split('.')[0]=='dlinfer']}

    def norm_perf_mode(self, mode, diagnostic=False):
        assert mode in ('old', 'new')
        runner = self._norm_runner
        with self.model_agent.all_context(), torch.inference_mode():
            torch.npu.synchronize()
            runner._perf_enabled = False
            assert not runner._perf_spans, 'Collect previous diagnostic spans first'
            for name, old, new in self._norm_layers:
                runner.model.set_submodule(name, old if mode == 'old' else new)
            runner._runner_map, runner.graph_pool_handle = self._norm_modes[mode]
            runner._perf_enabled = diagnostic
            self._norm_mode = mode
        return {'rank': self.rank, 'pid': os.getpid(), 'mode': mode,
                'graph_keys': list(runner._runner_map), 'diagnostic': diagnostic}

    def norm_perf_collect(self):
        runner = self._norm_runner
        with self.model_agent.all_context():
            torch.npu.synchronize()
            spans = [dict(row, device_ms=start.elapsed_time(end)) for start, end, row in runner._perf_spans]
            runner._perf_spans = []
            runner._perf_enabled = False
        return {'rank': self.rank, 'pid': os.getpid(), 'mode': self._norm_mode,
                'graph_keys': list(runner._runner_map), 'spans': spans}
```
