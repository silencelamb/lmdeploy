# gdn_state_repro_v2.py

```python
"""Fixed-input/state GDN replay. No model, Ray, HCCL or sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent import fused_recurrent_gated_delta_rule

def sha(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--tokens',type=int,default=4)
    p.add_argument('--graph',action='store_true')
    args=p.parse_args();assert not args.output.exists(),args.output
    torch.set_num_threads(1);torch.manual_seed(42);torch.npu.manual_seed_all(42)
    dev='npu:0';T=args.tokens;H=8;HV=16;D=128
    q=torch.randn((1,T,H,D),dtype=torch.bfloat16).to(dev)
    k=torch.randn_like(q);v=torch.randn((1,T,HV,D),dtype=torch.bfloat16).to(dev)
    g=-torch.rand((1,T,HV),dtype=torch.float32).to(dev)
    beta=torch.rand((1,T,HV),dtype=torch.bfloat16).to(dev)
    initial=torch.randn((8,HV,D,D),dtype=torch.float32).to(dev)*.01
    state=initial.clone()
    cu=torch.tensor([0,T],dtype=torch.int32,device=dev)
    ids=torch.tensor([1],dtype=torch.int64,device=dev)
    lengths=torch.tensor([27],dtype=torch.int32,device=dev)
    def call():
        return fused_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=state,
            inplace_final_state=True,cu_seqlens=cu,cache_seqlens_rb=lengths,
            state_ids_rb=ids,num_state=4,use_qk_l2norm_in_kernel=True)[0]
    call();torch.npu.synchronize()
    graph=None
    if args.graph:
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,auto_dispatch_capture=True):
            captured_out=call()
        torch.npu.synchronize()
    outputs=[];ref_out=None;ref_state=None
    for i in range(args.iterations):
        state.copy_(initial)
        if graph:
            graph.replay();out=captured_out
        else:out=call()
        torch.npu.synchronize()
        oc=out.cpu();sc=state.cpu()
        if i==0:
            ref_out=oc.clone();ref_state=sc.clone()
            torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),g=g.cpu(),beta=beta.cpu(),
                initial=initial.cpu(),output=ref_out,state=ref_state),args.output.with_suffix('.pt'))
        row=dict(iteration=i,output_mismatch=int((oc!=ref_out).sum()),state_mismatch=int((sc!=ref_state).sum()),
                 output_max_abs=float((oc.float()-ref_out.float()).abs().max()),
                 state_max_abs=float((sc-ref_state).abs().max()),state_sha=sha(sc))
        if row['state_mismatch']:
            row['mismatch_per_slot']=(sc!=ref_state).flatten(1).sum(1).tolist()
            bad=args.output.with_name(args.output.stem+'-first-bad.pt')
            if not bad.exists():torch.save(dict(iteration=i,output=oc,state=sc),bad)
        outputs.append(row)
    result=dict(options=vars(args)|{'output':str(args.output)},seed=42,
        input_hashes={n:sha(t) for n,t in dict(q=q,k=k,v=v,g=g,beta=beta,initial=initial).items()},
        torch=torch.__version__,torch_npu=torch_npu.__version__,results=outputs)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('REPLAY',args.graph,'output_mismatch_runs',sum(x['output_mismatch']>0 for x in outputs),
          'state_mismatch_runs',sum(x['state_mismatch']>0 for x in outputs),
          'max_state_diff',max(x['state_max_abs'] for x in outputs),flush=True)

if __name__=='__main__':main()
```
