"""Interleaved graph microbenchmark; correctness is reported separately.

This measures the GDN kernel only. It does not measure serving throughput or
pretend the original corrupted-state implementation is a valid accuracy baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu
import lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent as module


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=8);p.add_argument('--iterations',type=int,default=500)
    a=p.parse_args();assert not a.output.exists(),a.output
    torch.set_num_threads(1);torch.manual_seed(42);torch.npu.manual_seed_all(42)
    fpath=Path('/root/workspace/a3-mtp-ray-investigation-20261116/results/gdn-real-fixed-graph-r0.pt')
    fixture=torch.load(fpath,weights_only=True);device='npu:0'
    current=3;kernel=module.fused_recurrent_gated_delta_rule_fwd_kernel
    class Options:
        def __getitem__(self,grid):
            launch=kernel[grid]
            def run(*args,**kwargs):kwargs['num_stages']=current;return launch(*args,**kwargs)
            return run
    module.fused_recurrent_gated_delta_rule_fwd_kernel=Options()
    rows=[];data=dict(source=module.__file__,fixture=str(fpath),fixture_sha256=hashlib.sha256(fpath.read_bytes()).hexdigest(),
        versions=dict(torch=torch.__version__,torch_npu=torch_npu.__version__),affinity=sorted(os.sched_getaffinity(0)),
        options=vars(a)|{'output':str(a.output)},cases=rows,complete=False)
    def save():a.output.write_text(json.dumps(data,indent=2)+'\n')
    save()
    for batch in (1,4,16):
        for queries in (2,4):
            slots=queries;HV=16;D=128
            inputs={name:fixture[name][:,:queries].repeat(1,batch,*([1]*(fixture[name].ndim-2))).to(device)
                    for name in ('q','k','v','g','beta')}
            initial=torch.zeros((batch+1)*slots,HV,D,D,dtype=torch.float32,device=device)
            expected=torch.zeros_like(initial,device='cpu')
            for i in range(1,batch+1):
                initial[i*slots+27%slots]=fixture['initial'][7].to(device)
                for t in range(queries):expected[i*slots+(27+t+1)%slots]=fixture['state'][4+t]
            state=initial.clone();cu=torch.arange(batch+1,dtype=torch.int32,device=device)*queries
            ids=torch.arange(1,batch+1,dtype=torch.int64,device=device)
            lengths=torch.full((batch,),27,dtype=torch.int32,device=device)
            def call():
                return module.fused_recurrent_gated_delta_rule(**inputs,initial_state=state,inplace_final_state=True,
                    cu_seqlens=cu,cache_seqlens_rb=lengths,state_ids_rb=ids,num_state=slots,use_qk_l2norm_in_kernel=True)[0]
            graphs={};outs={};checks={}
            for stages in (3,1):
                current=stages;state.copy_(initial);call();torch.npu.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,auto_dispatch_capture=True):out=call()
                torch.npu.synchronize();graphs[stages]=graph;outs[stages]=out
                state.copy_(initial);graph.replay();torch.npu.synchronize()
                actual=state.cpu()
                checks[stages]=dict(state_close=bool(torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)),
                    max_abs=float((actual-expected).abs().max()))
            row=dict(batch=batch,queries=queries,states=slots,correctness=checks,measurements=[]);rows.append(row)
            for round_id in range(a.rounds):
                order=(3,1) if round_id%2==0 else (1,3)
                for stages in order:
                    graph=graphs[stages];state.copy_(initial)
                    for _ in range(10):graph.replay()
                    torch.npu.synchronize()
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(a.iterations):graph.replay()
                    end.record();torch.npu.synchronize()
                    row['measurements'].append(dict(round=round_id,num_stages=stages,device_ms=start.elapsed_time(end)/a.iterations))
                save()
            original=[x['device_ms'] for x in row['measurements'] if x['num_stages']==3]
            fixed=[x['device_ms'] for x in row['measurements'] if x['num_stages']==1]
            row['summary']=dict(original_median=statistics.median(original),fixed_median=statistics.median(fixed),
                original_range=[min(original),max(original)],fixed_range=[min(fixed),max(fixed)],
                median_change_pct=100*(statistics.median(fixed)/statistics.median(original)-1),
                paired_change_pct=[100*(y/x-1) for x,y in zip(original,fixed)])
            print('PERF2',batch,queries,row['summary'],flush=True)
            for graph in graphs.values():graph.reset()
            del graphs,outs,state,initial,inputs
    data['complete']=True;save()


if __name__=='__main__':main()
