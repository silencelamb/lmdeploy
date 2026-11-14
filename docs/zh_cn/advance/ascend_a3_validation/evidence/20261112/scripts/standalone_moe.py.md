# standalone_moe.py

原始验证文件快照；源码 SHA256：`f150b7dd620681279ebb2219840b93217b619ce87c8ae2400e5ec8e4d1edc09c`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Real NPU >1024-expert eager/capture/replay against an independent CPU reference."""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

assert importlib.util.find_spec('lmdeploy') is None
class NoLMDeploy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0]=='lmdeploy':
            raise AssertionError('Standalone vendor op imported '+fullname)
sys.meta_path.insert(0,NoLMDeploy())
import torch
import torch_npu
from dlinfer.vendor.ascend import moe

torch.manual_seed(20261112)
E,H,I,N=1056,64,64,32
up=(torch.randn(E,2*I,H)*.08).bfloat16()
down=(torch.randn(E,H,I)*.08).bfloat16()
x=(torch.randn(N,H)*.5).bfloat16()
counts=torch.zeros(E,dtype=torch.int64)
counts[[0,1,511,1021,1022,1030,1054,1055]]=4

def reference(x,counts):
    rows=[];offset=0
    for expert,count in enumerate(counts.tolist()):
        if count:
            z=(x[offset:offset+count].float()@up[expert].float().T).bfloat16()
            gate,value=z.chunk(2,dim=-1)
            activated=(torch.nn.functional.silu(gate.float())*value.float()).bfloat16()
            rows.append((activated.float()@down[expert].float().T).bfloat16())
            offset+=count
    return torch.cat(rows).float()

xn,un,dn,cn=x.npu(),up.npu(),down.npu(),counts.npu()
results=[]
def check(label,actual,expected):
    actual=actual.cpu().float()
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.008)
    results.append(dict(label=label,max_abs=(actual-expected).abs().max().item()))

for catchall in [False,True]:
    moe._MOE_PREFILL_USE_CATCHALL=catchall
    for group_type in [0,1]:
        gl=cn.cumsum(0) if group_type==0 else cn
        check(f'eager-catchall{catchall}-type{group_type}',moe.apply_mlp(xn,un,dn,gl,group_type),reference(x,counts))
moe._MOE_PREFILL_USE_CATCHALL=False
moe.apply_mlp(xn,un,dn,cn,1)
torch.npu.synchronize()
g=torch.npu.NPUGraph()
stream=torch.npu.Stream()
try:
    with torch.npu.graph(g,stream=stream,auto_dispatch_capture=True):
        output=moe.apply_mlp(xn,un,dn,cn,1)
    g.replay();torch.npu.synchronize()
    check('graph-initial',output,reference(x,counts))
    # Change both values and expert counts without changing any captured shapes.
    next_x=(x.float()*.7+.1).bfloat16()
    next_counts=counts.roll(1)
    xn.copy_(next_x);cn.copy_(next_counts)
    g.replay();torch.npu.synchronize()
    check('graph-replay-changed-routing',output,reference(next_x,next_counts))
finally:
    g.reset()
assert not any(n.split('.')[0]=='lmdeploy' for n in sys.modules)
root=Path(os.environ['VALIDATION_ROOT'])
(root/'results/standalone-moe.json').write_text(json.dumps(dict(experts=E,rows=N,results=results,lmdeploy_imported=False),indent=2))
print(json.dumps(results,indent=2))

```
