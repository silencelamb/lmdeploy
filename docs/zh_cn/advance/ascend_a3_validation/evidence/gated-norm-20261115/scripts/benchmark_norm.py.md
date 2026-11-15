# benchmark_norm.py

```python
"""Alternating same-process old/new operator timing, including graph replay."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from lmdeploy.pytorch.backends.ascend.runtime import initialize
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.nn import RMSNormGated

ROOT=Path(__file__).resolve().parents[1]
initialize()
spec=importlib.util.spec_from_file_location('timing_frozen_norm',ROOT/'lmdeploy-before/lmdeploy/pytorch/kernels/ascend/triton_ops/rms_norm_gated.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
results=[]
with torch.inference_mode(), get_device_manager().context(DeviceContext('ascend')):
    for rows in (32,128,5440):
        x,z=(torch.randn(rows,128,device='npu',dtype=torch.bfloat16) for _ in range(2))
        before=old.RMSNormGated(128,eps=1e-6,norm_before_gate=True,device='npu')
        after=RMSNormGated(128,eps=1e-6,device='npu',dtype=torch.bfloat16)
        functions=dict(before=lambda:before(x,z),after=lambda:after(x,z))
        for fn in functions.values():
            for _ in range(10):fn()
        torch.npu.synchronize()
        for mode in ('eager','graph'):
            funcs=functions.copy()
            graphs=[]
            if mode=='graph':
                for key,fn in functions.items():
                    graph=torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(32):fn()
                    graphs.append(graph);funcs[key]=graph.replay
            samples=dict(before=[],after=[])
            device_samples=dict(before=[],after=[])
            count=100 if mode=='eager' else 30
            factor=1 if mode=='eager' else 32
            for rep in range(10):
                order=('before','after') if rep%2==0 else ('after','before')
                for key in order:
                    torch.npu.synchronize()
                    start,end=(torch.npu.Event(enable_timing=True) for _ in range(2))
                    began=time.perf_counter();start.record()
                    for _ in range(count):funcs[key]()
                    end.record();end.synchronize()
                    samples[key].append((time.perf_counter()-began)*1e6/count/factor)
                    device_samples[key].append(start.elapsed_time(end)*1000/count/factor)
            med={key:statistics.median(value) for key,value in samples.items()}
            dev={key:statistics.median(value) for key,value in device_samples.items()}
            row=dict(rows=rows,width=128,mode=mode,wall_us=med,device_us=dev,samples_wall_us=samples,
                     samples_device_us=device_samples,wall_change_pct=(med['after']/med['before']-1)*100,
                     device_change_pct=(dev['after']/dev['before']-1)*100)
            results.append(row);print(json.dumps({k:v for k,v in row.items() if not k.startswith('samples')}),flush=True)
(ROOT/'results/norm-performance.json').write_text(json.dumps(results,indent=2)+'\n')
```
