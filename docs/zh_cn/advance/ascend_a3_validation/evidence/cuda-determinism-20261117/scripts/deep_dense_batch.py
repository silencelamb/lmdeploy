"""Profile native BF16 GEMM and explicit precision controls on real frozen rows."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
def exact(x,y):return x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def reqmap(t):
    p=t['_inputs'];m=p['attn_metadata'];ids=p['input_ids'].flatten()
    return {tuple(ids[s:s+n].tolist()):(s,n) for s,n in zip(m['q_start_loc'].tolist(),m['q_seqlens'].tolist())}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    rows=[];old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with torch.inference_mode():
        for model,key in [('flash','model.layers.0.self_attn.fg_a_proj'),('s2','model.language_model.layers.0.mlp.gate')]:
            ts=[torch.load(ROOT/'results'/f'diagnostic-freeze-r1-pilot-perf-{model}-http-pair{pair}-pr-f64-frozen/rank0-first.pt',map_location='cpu',weights_only=False,mmap=True) for pair in [0,1]]
            mappings=[reqmap(t) for t in ts];common=set(mappings[0])&set(mappings[1]);weight=ts[0][key+'.direct_parameters']['weight'].cuda()
            assert exact(ts[0][key+'.direct_parameters']['weight'],ts[1][key+'.direct_parameters']['weight'])
            inputs=[t[key+'.in'][0].reshape(-1,weight.shape[1]).cuda() for t in ts]
            row=dict(model=model,input_rows=[x.shape[0] for x in inputs],weight_shape=list(weight.shape),common_requests=len(common),original_bf16_reduced_precision_reduction=old,
                     allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),controls={},native_matches_captured=[])
            for mode in ['native','disable_reduced_precision_reduction','fp32_compute_then_bf16']:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False if mode=='disable_reduced_precision_reduction' else old)
                outs=[]
                for i,x in enumerate(inputs):
                    xx,ww=(x.float(),weight.float()) if mode=='fp32_compute_then_bf16' else (x,weight)
                    for _ in range(2):F.linear(xx,ww)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:out=F.linear(xx,ww).to(x.dtype)
                    prof.export_chrome_trace(str(dest/f'{model}-{mode}-M{x.shape[0]}.json'));outs.append(out.cpu())
                    if mode=='native':
                        target=ts[i][key+'.out']
                        if model=='flash':matched=exact(out.cpu(),target.reshape_as(out))
                        else:
                            probabilities=F.softmax(out,dtype=torch.float,dim=-1)
                            top,indices=torch.topk(probabilities,target[1].shape[-1],dim=-1);top/=top.sum(-1,keepdim=True)
                            matched=all(exact(v.cpu(),w) for v,w in zip([probabilities,top,indices],target))
                        row['native_matches_captured'].append(matched);assert matched,(model,i)
                checks=[]
                for ids in common:
                    (s,n),(u,m)=[v[ids] for v in mappings];assert n==m
                    aa,bb=outs[0][s:s+n],outs[1][u:u+n]
                    checks.append(dict(input_rows_equal=exact(inputs[0][s:s+n],inputs[1][u:u+n]),output_equal=exact(aa,bb),different_elements=int((aa!=bb).sum()),max_abs=float((aa.float()-bb.float()).abs().max())))
                row['controls'][mode]=checks
                print(model,mode,checks,flush=True)
            rows.append(row);del ts,weight,inputs
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old
    (dest/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
