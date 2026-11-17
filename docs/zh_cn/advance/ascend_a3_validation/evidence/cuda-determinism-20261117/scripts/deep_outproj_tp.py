"""Full TP8 frozen out_proj decomposition. No full-model acceptance claims."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT=Path(__file__).resolve().parents[1]
MODULE='model.language_model.layers.0.linear_attn.out_proj'
def exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8),b.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8))
def details(a,b):
    same=exact(a,b);out=dict(equal=same,shape=list(a.shape),dtype=str(a.dtype))
    if not same:
        out.update(unequal_elements=int((a!=b).sum()),max_abs=float((a.float()-b.float()).abs().max()))
    return out
def keymap(d):
    keys={}
    for pre in d['prefills']:
        p=pre['inputs'];m=p['attn_metadata'];ids=p['input_ids'].flatten()
        for i,(s,n) in enumerate(zip(m['q_start_loc'].tolist(),m['q_seqlens'].tolist())):
            keys[int(p['state_ids'][i])]=tuple(ids[s:s+n].tolist())
    return [keys[int(x)] for x in d['decode_inputs']['state_ids'].tolist()]
def worker(rank,port,output):
    torch.set_num_threads(2);torch.cuda.set_device(rank);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    outdir=Path(output);outdir.mkdir(parents=True,exist_ok=True)
    # Every A rank has a valid first-decode capture. B is constructed by an
    # explicitly recorded row permutation, never by filling missing traces.
    path=ROOT/'results/decode-order-r2-base'/f'orderA-rank{rank}.pt'
    data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    b0=torch.load(ROOT/'results/decode-order-r2-base/orderB-rank0.pt',map_location='cpu',weights_only=False,mmap=True)
    akeys=keymap(data);bkeys=keymap(b0);perm=[akeys.index(k) for k in bkeys]
    x=data['layer_buffers'][MODULE+'.in'][0].cuda();captured_a=data['layer_buffers'][MODULE+'.out'].cuda()
    captured_b=b0['layer_buffers'][MODULE+'.out'].cuda()
    wpath=ROOT/'results'/f'diagnostic-freeze-r1-pilot-perf-s2-http-pair1-base-frozen/rank{rank}-first.pt'
    weights=torch.load(wpath,map_location='cpu',weights_only=False,mmap=True)[MODULE+'.direct_parameters']
    weight=weights['weight'].cuda();scale=weights['weight_scale_inv'].cuda()
    from lmdeploy.pytorch.backends import get_backend
    from lmdeploy.pytorch.backends.blockedf8_modules import LinearBlockedF8BuildSpec
    spec=LinearBlockedF8BuildSpec(weight.shape[1],weight.shape[0],128,False,x.dtype,weight.dtype,'ue8m0')
    impl=get_backend().build_op(spec,enable_deterministic=False)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=8,
                            timeout=datetime.timedelta(seconds=90))
    result=dict(rank=rank,permutation=perm,implementation=type(impl).__name__,scale_fmt=impl.scale_fmt,
                x_shape=list(x.shape),x_stride=list(x.stride()),weight_shape=list(weight.shape),
                weight_dtype=str(weight.dtype),torch=torch.__version__,nccl=torch.cuda.nccl.version(),
                experiment='Both matrix products are from the actual unchanged CUDA provider. Constructed B is P(actual A) on all8ranks. Separate from historical missing-B captures.',comparisons={})
    tensors={}
    try:
        with torch.inference_mode():
            pidx=torch.tensor(perm,device=x.device)
            xb=x.index_select(1,pidx)
            for _ in range(3):impl.forward(x,weight,scale,all_reduce=False)
            la=impl.forward(x,weight,scale,all_reduce=False)
            lb=impl.forward(xb,weight,scale,all_reduce=False)
            expected_local_b=la.index_select(1,pidx)
            result['comparisons']['local_gemm_permutation']=details(lb,expected_local_b)
            result['comparisons']['local_gemm_repeat']=details(la,impl.forward(x,weight,scale,all_reduce=False))
            tensors.update(xA=x.cpu(),xB=xb.cpu(),localA=la.cpu(),localB=lb.cpu())
            def reduce(v):
                buf=v.clone();dist.all_reduce(buf,group=dist.group.WORLD);return buf
            ra=reduce(la);rb=reduce(lb);rb_fixed=reduce(expected_local_b)
            result['comparisons']['eager_matches_captured_A']=details(ra,captured_a)
            result['comparisons']['eager_matches_captured_B_rank0']=details(rb,captured_b)
            result['comparisons']['eager_reduction_permutation']=details(rb,ra.index_select(1,pidx))
            result['comparisons']['reduction_only_permutation']=details(rb_fixed,ra.index_select(1,pidx))
            result['comparisons']['eager_reduce_repeat']=details(ra,reduce(la))
            tensors.update(reducedA=ra.cpu(),reducedB=rb.cpu(),capturedA=captured_a.cpu(),capturedB_rank0=captured_b.cpu())
            # Same collective inside a real CUDA graph, as in the original path.
            buf=la.clone();side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):buf.copy_(la);dist.all_reduce(buf)
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();dist.barrier()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):dist.all_reduce(buf)
            graphs=[]
            for v in [la,lb,expected_local_b,la]:
                buf.copy_(v);g.replay();graphs.append(buf.clone())
            ga,gb,gfixed,gagain=graphs
            result['comparisons']['graph_matches_captured_A']=details(ga,captured_a)
            result['comparisons']['graph_matches_captured_B_rank0']=details(gb,captured_b)
            result['comparisons']['graph_reduction_permutation']=details(gb,ga.index_select(1,pidx))
            result['comparisons']['graph_reduction_only_permutation']=details(gfixed,ga.index_select(1,pidx))
            result['comparisons']['graph_repeat']=details(ga,gagain)
            tensors.update(graphA=ga.cpu(),graphB=gb.cpu())
            # Counterfactual precision control only; never replaces native output.
            fa=reduce(la.float()).to(la.dtype);fb=reduce(lb.float()).to(lb.dtype)
            result['comparisons']['fp32_reduction_control_permutation']=details(fb,fa.index_select(1,pidx))
            tensors.update(fp32reduceA=fa.cpu(),fp32reduceB=fb.cpu())
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                impl.forward(x,weight,scale,all_reduce=True,group=dist.group.WORLD)
            prof.export_chrome_trace(str(outdir/f'rank{rank}-trace.json'))
            del g,graphs
            torch.cuda.synchronize();dist.barrier()
            torch.save(tensors,outdir/f'rank{rank}-outputs.pt')
            (outdir/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
            print('RANK',rank,{k:v['equal'] for k,v in result['comparisons'].items()},flush=True)
    finally:
        dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,a.output),nprocs=8,join=True)
if __name__=='__main__':main()
