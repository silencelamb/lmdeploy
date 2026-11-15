"""Frozen first target-step timing; restores state around diagnostic replays.

Not an end-to-end throughput benchmark: every measured call uses identical token
IDs, positions and active cache state. State restore is outside device events.
"""
import hashlib
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
            saved=[]
            for i,cache in enumerate(kwargs['past_key_values']):
                ids=(kwargs['state_ids'] if layer_types[i]=='linear_attention'
                     else kwargs['attn_metadata'].block_offsets[0,:1].long())
                for value in cache[:2]:saved.append((value,ids,value.index_select(0,ids).clone()))
            torch.npu.synchronize()
            before_hash=digest([x[2] for x in saved])
            result=original(runner,**kwargs);torch.npu.synchronize()
            original_result_hash=digest(result)
            post=[x.index_select(0,ids).clone() for x,ids,_ in saved]
            def restore():
                for value,ids,copy in saved:value.index_copy_(0,ids,copy)
                torch.npu.synchronize()
            times=[]
            for i in range(48):
                restore()
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                start.record();result=original(runner,**kwargs);end.record();torch.npu.synchronize()
                times.append(start.elapsed_time(end))
            restore();restored_hash=digest([x.index_select(0,ids) for x,ids,_ in saved])
            assert before_hash==restored_hash
            result=original(runner,**kwargs);torch.npu.synchronize()
            last_hash=digest(result)
            for (value,ids,_),copy in zip(saved,post):value.index_copy_(0,ids,copy)
            torch.npu.synchronize()
            path=Path(os.environ['VALIDATION_ROOT'])/'results'/f'perf2-target-{os.environ["PERF2_LABEL"]}-rank{worker.rank}.json'
            assert not path.exists(),path
            metadata=kwargs['attn_metadata']
            row=dict(rank=worker.rank,input_ids=kwargs['input_ids'].cpu().tolist(),
                positions=kwargs['position_ids'].cpu().tolist(),history=metadata.cache_seqlens.cpu().tolist(),
                state_before_sha256=before_hash,restored_state_sha256=restored_hash,
                first_output_sha256=original_result_hash,last_output_sha256=last_hash,
                repeat_output_equal=last_hash==original_result_hash,device_ms=times,
                median_ms=statistics.median(times[8:]),min_ms=min(times[8:]),max_ms=max(times[8:]),
                note='First 8 warm iterations excluded. Active cache copies restored outside timing; all calls share one target history.')
            path.write_text(json.dumps(row,indent=2)+'\n')
            print('FROZEN_TARGET',worker.rank,row['median_ms'],row['repeat_output_equal'],flush=True)
            return result
        AscendGraphRunner.__call__=traced

    def start(self):
        self.measure_started=True
        return super().start()


if __name__=='__main__':
    from lmdeploy.pytorch.engine.executor import ray_executor
    from perf2_frozen_target import FrozenTargetWorker as ImportedWorker
    ray_executor.RayWorkerWrapper=ImportedWorker
    seed()
    from pipeline_case import main
    main()
