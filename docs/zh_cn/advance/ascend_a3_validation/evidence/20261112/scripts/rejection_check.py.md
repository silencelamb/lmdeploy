# rejection_check.py

原始验证文件快照；源码 SHA256：`aa0da504af79cc7ea5e2d3d894be45b47ac2a4c137da687518afcf22f5e3b774`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import argparse
import itertools
import json
import os
from pathlib import Path
import torch

from lmdeploy.utils import try_import_deeplink
try_import_deeplink('ascend')
if os.environ['VALIDATION_ENV']=='native':
    from lmdeploy.pytorch.backends.ascend.rejection_sampling import AscendRejectionSamplingImpl
else:
    from lmdeploy.pytorch.backends.dlinfer.ascend.rejection_sampling import AscendRejectionSamplingImpl
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--triton',action='store_true');a=p.parse_args()
if a.triton: os.environ['DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON']='1'
sampler=AscendRejectionSamplingImpl()
torch.manual_seed(42)
N,K,V=65536,3,4
prob=torch.tensor([.1,.2,.3,.4],device='npu',dtype=torch.float32)
logits=prob.log().expand(N,K,V).contiguous()
draft=torch.zeros((N,K),device='npu',dtype=torch.int64)
bonus=torch.full((N,),3,device='npu',dtype=torch.int64)
results=[]
for mixed in [False,True]:
    greedy=None if not mixed else torch.arange(N,device='npu')%2==0
    output,rejected,last=sampler.forward(logits,draft,bonus,is_greedy=greedy)
    sampled=output[:,0].cpu() if not mixed else output[1::2,0].cpu()
    freq=torch.bincount(sampled,minlength=V).float()/sampled.numel()
    assert (freq-prob.cpu()).abs().max()<.015, freq
    if mixed:
        assert (output[::2,0]==3).all().item()
        assert (rejected[::2]==K).all().item()
    results.append(dict(mixed=mixed,draws=sampled.numel(),first_token_frequencies=freq.tolist()))
patterns=list(itertools.product([0,1],repeat=K))
target=torch.tensor([[3,3,3,2]]*len(patterns),device='npu')
draft=torch.tensor([[3 if ok else 0 for ok in row] for row in patterns],device='npu')
output,rejected,last=sampler.forward_greedy(target,draft)
for i,row in enumerate(patterns):
    accepted_prefix=next((j for j,v in enumerate(row) if not v),K)
    expected=target[i,:accepted_prefix+1].cpu().tolist()
    assert output[i,:accepted_prefix+1].cpu().tolist()==expected
    assert rejected[i].item()==K-accepted_prefix
    assert last[i].item()==expected[-1]
Path(a.output).write_text(json.dumps(dict(implementation='triton' if a.triton else 'torch',random_cases=results,greedy_patterns=len(patterns)),indent=2))
print('PASS',a.output)

```
