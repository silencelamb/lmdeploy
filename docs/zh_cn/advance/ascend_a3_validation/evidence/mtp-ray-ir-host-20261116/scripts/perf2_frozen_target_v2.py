"""Frozen first target-step timing; restores state around diagnostic replays.

Not an end-to-end throughput benchmark: every measured call uses identical token
IDs, positions and active cache state. State restore is outside device events.
"""
import hashlib
import copy as copy_module
import json
import os
from pathlib import Path
import statistics

import numpy as np
import random
import torch
import torch_npu
from lmdeploy.pytorch.engine.executor.ray_executor import RayWorkerWrapper


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.npu.manual_seed_all(42)


def digest(values):
    h=hashlib.sha256()
    if isinstance(values,dict):values=list(values.values())
    for x in values:
        if isinstance(x,torch.Tensor):h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x,(list,tuple,dict)):h.update(digest(x).encode())
    return h.hexdigest()


def clone_data(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:clone_data(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(clone_data(v) for v in value)
    return value


def clone_metadata(value):
    out=copy_module.copy(value)
    for k,v in vars(value).items():setattr(out,k,clone_data(v))
    return out


class FrozenTargetWorker(RayWorkerWrapper):
    def build_model(self):
        seed();super().build_model();self.measure_started=False;self.measured=False
        model=self.model_agent.patched_model
        layer_types=model.model.language_model.config.layer_types
        from lmdeploy.pytorch.backends.ascend.graph_runner import AscendGraphRunner
        original=AscendGraphRunner.__call__
        worker=self
        def traced(runner,**kwargs):
            if (runner.model is not model or not worker.measure_started or worker.measured
                    or not kwargs['attn_metadata'].is_multi_token_decoding):
                return original(runner,**kwargs)
            worker.measured=True
            assert kwargs['input_ids'].numel()==4
            from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
            manager=get_step_ctx_manager();context=manager.current_context()
            context_template=copy_module.copy(context)
            for k,v in vars(context).items():
                if isinstance(v,torch.Tensor):setattr(context_template,k,v.clone())
            context_template.moe_metadata=clone_metadata(context.moe_metadata)
            metadata_template=clone_metadata(kwargs['attn_metadata'])
            kwargs_template={k:(clone_data(v) if k not in ('past_key_values','attn_metadata','moe_metadata') else v) for k,v in kwargs.items()}
            def replay():
                step=copy_module.copy(context_template)
                step.moe_metadata=copy_module.copy(context_template.moe_metadata)
                call_kwargs=dict(kwargs_template);call_kwargs['attn_metadata']=copy_module.copy(metadata_template)
                call_kwargs['moe_metadata']=step.moe_metadata
                step.attn_metadata=call_kwargs['attn_metadata']
                with manager.context(step):return original(runner,**call_kwargs)
            saved=[]
            for i,cache in enumerate(kwargs['past_key_values']):
                ids=(kwargs['state_ids'] if layer_types[i]=='linear_attention'
                     else kwargs['attn_metadata'].block_offsets[0,:1].long())
                for value in cache[:2]:saved.append((value,ids,value.index_select(0,ids).clone()))
            torch.npu.synchronize()
            before_hash=digest([x[2] for x in saved])
            result=original(runner,**kwargs);torch.npu.synchronize()
            first_result_copy=clone_data(result)
            original_result_hash=digest(result)
            post=[x.index_select(0,ids).clone() for x,ids,_ in saved]
            def restore():
                for value,ids,copy in saved:value.index_copy_(0,ids,copy)
                torch.npu.synchronize()
            restore();reference=replay();torch.npu.synchronize();reference_hash=digest(reference)
            times=[];output_hashes=[]
            for i in range(48):
                restore()
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                start.record();result=replay();end.record();torch.npu.synchronize()
                times.append(start.elapsed_time(end))
                output_hashes.append(digest(result))
                assert output_hashes[-1]==reference_hash,'Frozen steady target output changed'
            restore();restored_hash=digest([x.index_select(0,ids) for x,ids,_ in saved])
            assert before_hash==restored_hash
            result=replay();torch.npu.synchronize()
            last_hash=digest(result)
            for (value,ids,_),copy in zip(saved,post):value.index_copy_(0,ids,copy)
            torch.npu.synchronize()
            path=Path(os.environ['VALIDATION_ROOT'])/'results'/f'perf2-target-{os.environ["PERF2_LABEL"]}-rank{worker.rank}.json'
            assert not path.exists(),path
            metadata=kwargs['attn_metadata']
            row=dict(rank=worker.rank,input_ids=kwargs['input_ids'].cpu().tolist(),
                positions=kwargs['position_ids'].cpu().tolist(),history=metadata.cache_seqlens.cpu().tolist(),
                state_before_sha256=before_hash,restored_state_sha256=restored_hash,
                setup_output_sha256=original_result_hash,first_output_sha256=reference_hash,last_output_sha256=last_hash,
                repeat_output_equal=last_hash==reference_hash,device_ms=times,output_hashes=output_hashes,
                median_ms=statistics.median(times[8:]),min_ms=min(times[8:]),max_ms=max(times[8:]),
                note='First 8 warm iterations excluded. Active cache copies restored outside timing; all calls share one target history.')
            path.write_text(json.dumps(row,indent=2)+'\n')
            print('FROZEN_TARGET',worker.rank,row['median_ms'],row['repeat_output_equal'],flush=True)
            return first_result_copy
        AscendGraphRunner.__call__=traced

    def start(self):
        self.measure_started=True
        return super().start()


if __name__=='__main__':
    from lmdeploy.pytorch.engine.executor import ray_executor
    from perf2_frozen_target_v2 import FrozenTargetWorker as ImportedWorker
    ray_executor.RayWorkerWrapper=ImportedWorker
    seed()
    from pipeline_case import main
    main()
