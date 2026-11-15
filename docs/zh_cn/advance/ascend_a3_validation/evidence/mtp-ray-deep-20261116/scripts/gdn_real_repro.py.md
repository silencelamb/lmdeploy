# gdn_real_repro.py

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
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0)
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
    # Reconstruct the exact first target GDN input from a layer snapshot. Only
    # A_log and dt_bias are loaded from the fixed checkpoint, using its TP split.
    from safetensors import safe_open
    import torch.nn.functional as F
    snap=torch.load(args.snapshot,weights_only=False)
    prefix='model.language_model.layers.0.linear_attn.'
    qkv=snap[prefix+'conv1d.out'].to(dev)
    q,k,v=[t.unflatten(-1,(-1,128)).contiguous() for t in qkv.split([1024,1024,2048],dim=-1)]
    b,act=snap[prefix+'in_proj_ba.out'].to(dev).chunk(2,dim=-1)
    index=json.loads(Path('/afs-weights/Sage-A1.5/model.safetensors.index.json').read_text())['weight_map']
    params={}
    for name in ['A_log','dt_bias']:
        path=Path('/afs-weights/Sage-A1.5')/index[prefix+name]
        with safe_open(str(path),framework='pt',device='cpu') as f:
            params[name]=f.get_tensor(prefix+name).chunk(2,dim=0)[args.rank].float().to(dev)
    beta=b.sigmoid();g=-params['A_log'].exp()*F.softplus(act.float()+params['dt_bias'])
    initial=torch.zeros_like(initial)
    initial[4:8]=snap['model.language_model.layers.0.cache_before'][1][0].to(dev)
    state=initial.clone()
    lengths=snap['_meta']['cache_seqlens'].to(dev)
    cu=snap['_meta']['cu_seqlens_q'].to(dev)
    assert lengths.item()==27 and q.shape[1]==4
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
    result=dict(options=vars(args)|{'output':str(args.output),'snapshot':str(args.snapshot)},seed=42,
        parameter_hashes={n:sha(t) for n,t in params.items()},
        input_hashes={n:sha(t) for n,t in dict(q=q,k=k,v=v,g=g,beta=beta,initial=initial).items()},
        torch=torch.__version__,torch_npu=torch_npu.__version__,results=outputs)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('REPLAY',args.graph,'output_mismatch_runs',sum(x['output_mismatch']>0 for x in outputs),
          'state_mismatch_runs',sum(x['state_mismatch']>0 for x in outputs),
          'max_state_diff',max(x['state_max_abs'] for x in outputs),flush=True)

if __name__=='__main__':main()
```
