"""Diagnostic default/stable-emission controls on captured GLM inputs.

Deterministic=True is a separate counterfactual, never a replacement of the
original model result. No source or production configuration is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk,_sparse_index_topk_byte_radix_kernel
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import FlashMLASparseImpl

ROOT=Path(__file__).resolve().parents[1]
def digest(x):return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def eq(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    t=torch.load(ROOT/'results/glm-frozen-operator-base.pt',map_location='cpu',weights_only=False,mmap=True)
    sel=t['selector'];scores=sel['scores'].cuda();qs=sel['q_seqlens'].cuda();ks=sel['kv_seqlens'].cuda();k=sel['top_k']
    fields=t['attention_input']['args'][5]['fields'];meta=SimpleNamespace(q_seqlens=fields['q_seqlens'].cuda(),cu_seqlens_k=fields['cu_seqlens_k'].cuda())
    kv_len=int(fields['kv_seqlens'][0]);cache=t['model.layers.0.cache_after'][0].flatten(0,1)[:kv_len].contiguous().cuda()
    query=t['attention_input']['args'][0].cuda();params=t['attention_impl']
    impl=FlashMLASparseImpl(mla_index_topk=params['mla_index_topk'],mla_index_tail=params['mla_index_tail'],num_heads=params['num_heads'],head_size=params['head_size'],scale=params['scale'],num_kv_heads=params['num_kv_heads'],v_head_size=params['v_head_size'],causal=params['causal'])
    rows=[];original_set=sel['indices'].cuda().sort(-1).values
    with torch.inference_mode():
        for dataset in ['real','all_equal_tie_control']:
            sc=scores if dataset=='real' else torch.ones((1,4096),device='cuda',dtype=torch.float32)
            q=qs if dataset=='real' else torch.tensor([1],device='cuda',dtype=torch.int32)
            lengths=ks if dataset=='real' else torch.tensor([4096],device='cuda',dtype=torch.int32)
            for deterministic in [False,True]:
                def run():return sparse_index_topk(sc,q,lengths,k,deterministic=deterministic)
                for _ in range(2):run()
                _sparse_index_topk_byte_radix_kernel(k,-1,1024,deterministic).get_kernel_source()
                for graph_mode in [False,True]:
                    if graph_mode:
                        g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):out=run()
                    orders=[];sets=[];atts=[];real_sets=[]
                    for repeat in range(32):
                        if graph_mode:g.replay()
                        else:out=run()
                        orders.append(digest(out));sets.append(digest(out.sort(-1).values))
                        if dataset=='real':
                            real_sets.append(eq(out.sort(-1).values,original_set))
                            # Follow exactly the selected order, without sorting.
                            atts.append(digest(impl._prefill_sparse(query,cache,out,meta)))
                    row=dict(dataset=dataset,deterministic=deterministic,graph=graph_mode,repeats=32,
                        distinct_orders=len(set(orders)),distinct_sets=len(set(sets)),
                        distinct_attention_outputs=len(set(atts)) if atts else None,
                        matches_original_set=all(real_sets) if real_sets else None,
                        order_hashes=orders,set_hashes=sets,attention_hashes=atts)
                    rows.append(row);print({key:value for key,value in row.items() if not key.endswith('_hashes')},flush=True)
                    if graph_mode:del g
                source=_sparse_index_topk_byte_radix_kernel(k,-1,1024,deterministic).get_kernel_source()
                (dest/f'selector-deterministic-{int(deterministic)}.cu').write_text(source)
    (dest/'results.json').write_text(json.dumps(dict(source='Actual unmodified byte-radix selector; counterfactual mode labelled separately',rows=rows),indent=2))
if __name__=='__main__':main()
