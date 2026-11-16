"""Seeded single-request target tracing, diagnostic process only.

Hooks clone on device, including inside captured graphs. Host reads happen only
after the outer GraphRunner returns. These are NOT performance measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from mtp_audit import MTPAuditWorker


def seed_all():
    import torch_npu  # noqa: F401; backend autoload is deliberately disabled.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)


def tree_copy(value, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if cpu else value.detach().clone()
    if isinstance(value, dict):
        return {k: tree_copy(v, cpu) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [tree_copy(v, cpu) for v in value]
    if value is None or isinstance(value, (str,int,float,bool)):
        return value
    return type(value).__name__


class DeepTraceWorker(MTPAuditWorker):
    def build_model(self):
        seed_all()
        super().build_model()
        self.deep_started = False
        self.deep_count = 0
        root = Path(os.environ['VALIDATION_ROOT'])
        label = os.environ['VALIDATION_AUDIT_LABEL']
        self.deep_root = root/'artifacts'/f'mtp-deep-{label}'/f'rank{self.rank}'
        self.deep_root.mkdir(parents=True, exist_ok=False)
        (self.deep_root/'seed.json').write_text(json.dumps(dict(seed=42, rank=self.rank,
            cpu_rng=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            npu_rng=hashlib.sha256(torch.npu.get_rng_state().cpu().numpy().tobytes()).hexdigest())))
        if os.environ.get('MTP_DEEP_TRACE', 'none') == 'none':
            return
        model = self.model_agent.patched_model
        self.deep_model = model
        self.deep_buffers = {}
        self.deep_current = {}
        self.deep_enabled = False
        names = []
        for name, mod in model.named_modules():
            if ('.layers.' in name or name.endswith(('embed_tokens','norm'))):
                names.append((name, type(mod).__name__))
                def hook(m, args, kwargs, out, name=name):
                    if not self.deep_enabled:
                        return
                    self.deep_current[name+'.out'] = tree_copy(out)
                mod.register_forward_hook(hook, with_kwargs=True)
            if type(mod).__name__ in ('Qwen3_5DecoderLayer','Qwen3_5MoeDecoderLayer'):
                def before(m, args, kwargs, name=name):
                    if not self.deep_enabled:
                        return
                    meta = kwargs['gated_delta_meta']
                    attn = kwargs['attn_metadata']
                    # Single active request; gather only its state/cache blocks.
                    ids = meta.state_ids if m.layer_type == 'linear_attention' else attn.block_offsets[0,:1].long()
                    caches = kwargs['past_key_value'][:2]
                    self.deep_current[name+'.cache_before'] = [c.index_select(0,ids.flatten()).clone() for c in caches]
                def after(m, args, kwargs, out, name=name):
                    if not self.deep_enabled:
                        return
                    meta = kwargs['gated_delta_meta']
                    attn = kwargs['attn_metadata']
                    ids = meta.state_ids if m.layer_type == 'linear_attention' else attn.block_offsets[0,:1].long()
                    self.deep_current[name+'.cache_after'] = [c.index_select(0,ids.flatten()).clone() for c in kwargs['past_key_value'][:2]]
                mod.register_forward_pre_hook(before, with_kwargs=True)
                mod.register_forward_hook(after, with_kwargs=True)
        (self.deep_root/'modules.json').write_text(json.dumps(names,indent=2))
        from lmdeploy.pytorch.backends.ascend.graph_runner import AscendGraphRunner
        from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
        original = AscendGraphRunner.__call__
        worker = self
        def traced(runner, **kwargs):
            if runner.model is not model:
                return original(runner, **kwargs)
            context = get_step_ctx_manager().current_context()
            graph = context.global_is_decoding() and runner.enable_graph(**kwargs)
            key = (runner.get_graph_key(**kwargs)+model.get_cudagraph_extra_key(**kwargs)) if graph else ('eager',)
            worker.deep_current = worker.deep_buffers.setdefault(key,{})
            worker.deep_enabled = kwargs['input_ids'].numel() <= 128 and (worker.deep_started or (graph and key[0] == 1))
            record = worker.deep_started and worker.deep_count < int(os.environ.get('MTP_DEEP_STEPS','8'))
            if record:
                meta = tree_copy(vars(kwargs['attn_metadata']), cpu=True)
                ins = tree_copy({k:v for k,v in kwargs.items() if k not in ('past_key_values','attn_metadata','moe_metadata')}, cpu=True)
            result = original(runner, **kwargs)
            if record:
                torch.npu.synchronize()
                cpu = tree_copy(worker.deep_current, cpu=True)
                cpu.update(_inputs=ins, _meta=meta, _graph_key=key,
                           _output=tree_copy(result,cpu=True))
                torch.save(cpu,worker.deep_root/f'step{worker.deep_count:03d}.pt')
                print('MTP_DEEP_SAVED',worker.rank,worker.deep_count,key,flush=True)
                worker.deep_count += 1
            return result
        AscendGraphRunner.__call__ = traced

    def start(self):
        self.deep_started = True
        return super().start()


if __name__ == '__main__':
    from lmdeploy.pytorch.engine.executor import ray_executor
    ray_executor.RayWorkerWrapper = DeepTraceWorker
    seed_all()
    from pipeline_case import main
    main()
