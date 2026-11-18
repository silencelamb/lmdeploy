"""Frozen synthetic-input replay of the two same-key chunk-o binaries.

No model or production cache mutation. This is an operator diagnosis, not an
end-to-end acceptance replacement. A/B kernels use exactly the same addresses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    out=ROOT/'results'/f'{a.label}.json';assert not out.exists()
    cache=ROOT/'artifacts'/f'{a.label}-isolated-cache';cache.mkdir()
    os.environ['TRITON_CACHE_DIR']=str(cache)
    import numpy as np
    import torch
    import torch_npu
    from triton.compiler.compiler import CompiledKernel
    torch.npu.set_device(0);torch.manual_seed(42)
    audit=json.loads((ROOT/'results/chunk-o-cross-cohort-audit-r0.json').read_text())
    paths={side:Path(audit['norm-'+side]['path']) for side in ['before','after']}
    metas={side:json.loads(path.with_suffix('.json').read_text()) for side,path in paths.items()}
    assert metas['before']==metas['after']
    for ext in ['.ttir','.ttadapter']:
        assert paths['before'].with_suffix(ext).read_bytes()==paths['after'].with_suffix(ext).read_bytes()
    # Exact10 runtime parameters in the retained TTIR. The nine H/Hg/K/V/tile/
    # feature constexprs are already embedded in both binaries, not launch args.
    names=['q','k','v','h','g','o','cu_seqlens','chunk_offsets','scale','T']
    signature=dict(enumerate(['*bf16']*4+['*fp32','*bf16','*i32','*i64','fp32','i32']))
    source=SimpleNamespace(signature=signature,constants={},fn=SimpleNamespace(arg_names=names))
    kernels={}
    data={'label':a.label,'seed':42,'cache_override':str(cache),'metadata_identical':True,
          'ir_identical':True,'source_files':{},'cases':[],'complete':False,
          'tolerance':{'atol':.001,'rtol':.01},'performance_requires_bitwise_ab':True}
    def save():out.write_text(json.dumps(data,indent=2)+'\n')
    for side,path in paths.items():
        groups={path.with_suffix('.json').name:str(path.with_suffix('.json')),path.name:str(path)}
        kernels[side]=CompiledKernel(source,groups,metas[side]['hash'])
        kernels[side]._init_handles()
        data['source_files'][side]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'function':str(kernels[side].function),'module':str(kernels[side].module)}
    assert kernels['before'].function!=kernels['after'].function,'Loader reused one handle; A/B replay invalid'
    save()
    H,Hg,K,V,BT=16,8,128,128,64
    for lengths in [[43],[128],[170],[170,86]]:
        total=sum(lengths);N=len(lengths);offsets=[0]
        for size in lengths:offsets.append(offsets[-1]+(size+BT-1)//BT)
        cu=[0]
        for size in lengths:cu.append(cu[-1]+size)
        q=torch.randn(1,total,Hg,K,dtype=torch.float32)*.05
        k=torch.randn_like(q)*.05
        v=torch.randn(1,total,H,V,dtype=torch.float32)*.05
        h=torch.randn(offsets[-1],H,K,V,dtype=torch.float32)*.05
        g=torch.empty(1,H,total,dtype=torch.float32)
        for begin,size in zip(cu,lengths):
            for start in range(0,size,BT):
                n=min(BT,size-start);g[0,:,begin+start:begin+start+n]=-torch.rand(H,n).cumsum(-1)*.05
        q,k,v,h=[x.to(torch.bfloat16) for x in [q,k,v,h]]
        ref=torch.empty_like(v);scale=K**-.5
        for seq,(begin,size) in enumerate(zip(cu,lengths)):
            for chunk,start in enumerate(range(0,size,BT)):
                count=min(BT,size-start);idx=slice(begin+start,begin+start+count)
                for head in range(H):
                    qq=q[0,idx,head//(H//Hg)].float();kk=k[0,idx,head//(H//Hg)].float()
                    vv=v[0,idx,head].float();hh=h[offsets[seq]+chunk,head].float();gg=g[0,head,idx]
                    score=qq@kk.T
                    gate=(gg[:,None]-gg[None,:]).exp()
                    score=torch.tril(score*gate).to(torch.bfloat16).float()
                    value=(qq@hh)*gg.exp()[:,None]*scale+(score@vv)*scale
                    ref[0,idx,head]=value.to(torch.bfloat16)
        qn,kn,vn,hn,gn=[x.npu() for x in [q,k,v,h,g]];output=torch.empty_like(vn)
        cun=torch.tensor(cu,dtype=torch.int32,device='npu');on=torch.tensor(offsets,dtype=torch.int64,device='npu')
        args=[qn,kn,vn,hn,gn,output,cun,on,scale,total];grid=(1,N*H,1)
        launchers={side:kernel[grid] for side,kernel in kernels.items()}
        row={'sequence_lengths':lengths,'grid':grid,'correctness':{},'rounds':[]};data['cases'].append(row);save()
        values={}
        for side,launch in launchers.items():
            launch(*args);torch.npu.synchronize();values[side]=output.cpu()
            err=(values[side].float()-ref.float()).abs()
            row['correctness'][side]={'max_abs_error':err.max().item(),'finite':bool(torch.isfinite(values[side]).all())}
            save();torch.testing.assert_close(values[side],ref,atol=.001,rtol=.01)
        row['bitwise_ab']=torch.equal(values['before'],values['after']);save()
        assert row['bitwise_ab'],'Different numerical results; do not call speed comparison correctness-qualified'
        for launch in launchers.values():
            for _ in range(100):launch(*args)
        torch.npu.synchronize()
        graphs={}
        for side,launch in launchers.items():
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(100):launch(*args)
            graphs[side]=graph
        for graph in graphs.values():
            for _ in range(10):graph.replay()
        torch.npu.synchronize()
        for repeat in range(20):
            times={}
            for side in (['before','after'] if repeat%2==0 else ['after','before']):
                start,end=(torch.npu.Event(enable_timing=True) for _ in range(2))
                start.record();graphs[side].replay();end.record();end.synchronize()
                times[side]=start.elapsed_time(end)*1000/100
            row['rounds'].append({'repeat':repeat,'microseconds_per_call':times})
        changes=[(x['microseconds_per_call']['after']/x['microseconds_per_call']['before']-1)*100 for x in row['rounds']]
        row['paired_median_time_change_pct']=float(np.median(changes));save()
        print('BINARY_CASE',lengths,'equal',row['bitwise_ab'],'change_pct',row['paired_median_time_change_pct'],flush=True)
    data['complete']=True;save()


if __name__=='__main__':main()
