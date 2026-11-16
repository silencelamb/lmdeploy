"""Compare migrated prefill/L2Norm with the frozen external implementation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from triton_ascend_kernels.attention.fla import chunk_gated_delta_rule_fwd as reference
from triton_ascend_kernels.norm.l2norm import l2norm_fwd as reference_norm
from lmdeploy.pytorch.kernels.ascend.triton_ops.fla.prefill import chunk_gated_delta_rule_fwd as migrated
from lmdeploy.pytorch.kernels.ascend.triton_ops.fla.prefill.l2norm import l2norm_fwd as migrated_norm

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
torch.set_num_threads(1);torch.manual_seed(42);torch.npu.manual_seed_all(42)
rows=[]
def exact(x,y):return torch.equal(x.cpu().contiguous().view(torch.uint8),y.cpu().contiguous().view(torch.uint8))
def sha(x):return hashlib.sha256(x.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save():a.output.write_text(json.dumps(dict(rows=rows,complete=False),indent=2)+'\n')
save()
for dtype in (torch.bfloat16,torch.float32):
    for tokens in (1,65,257):
        x=torch.randn(1,tokens,4,128,dtype=dtype).to('npu')
        old=reference_norm(x);new=migrated_norm(x);torch.npu.synchronize()
        same=exact(old,new);rows.append(dict(kind='l2norm',tokens=tokens,dtype=str(dtype),bitwise_equal=same,sha256=sha(new)));save();assert same
        print('L2NORM',tokens,dtype,'exact',same,flush=True)
for lengths in ([27],[129],[17,79]):
    tokens=sum(lengths);heads=4;dim=128;value_dim=64
    q,k=[torch.randn(1,tokens,heads,dim,dtype=torch.bfloat16).to('npu') for _ in range(2)]
    q,k=reference_norm(q),reference_norm(k)
    v=torch.randn(1,tokens,heads,value_dim,dtype=torch.bfloat16).to('npu')
    g=(-torch.rand(1,tokens,heads,dtype=torch.float32)*.05).to('npu')
    beta=torch.rand(1,tokens,heads,dtype=torch.bfloat16).to('npu')
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32,device='npu')
    for initialized in (False,True):
        state=(torch.randn(len(lengths),heads,dim,value_dim,dtype=torch.float32)*.01).to('npu') if initialized else None
        kwargs=dict(q=q,k=k,v=v,g=g,beta=beta,scale=dim**-.5,initial_state=state,output_final_state=True,cu_seqlens=cu)
        old=reference(**kwargs);new=migrated(**kwargs);torch.npu.synchronize()
        matched=[exact(x,y) for x,y in zip(old,new)]
        rows.append(dict(kind='prefill',lengths=lengths,initial_state=initialized,output_and_state_bitwise_equal=matched,
                         hashes=[sha(x) for x in new]));save();assert all(matched)
        print('PREFILL',lengths,initialized,matched,flush=True)
a.output.write_text(json.dumps(dict(rows=rows,complete=True),indent=2)+'\n')
