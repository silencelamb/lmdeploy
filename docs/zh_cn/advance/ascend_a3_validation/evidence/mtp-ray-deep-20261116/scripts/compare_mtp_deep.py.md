# compare_mtp_deep.py

```python
"""Compare only matching target input histories, preserving hook execution order."""
import argparse
import json
from pathlib import Path
import torch

def flatten(x,prefix=''):
    if isinstance(x,dict):
        for k,v in x.items(): yield from flatten(v,prefix+'.'+str(k))
    elif isinstance(x,(list,tuple)):
        for k,v in enumerate(x): yield from flatten(v,prefix+'.'+str(k))
    else: yield prefix,x

def compare(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:
            return dict(shape_a=list(a.shape),shape_b=list(b.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype))
        mask=(a!=b)
        if not mask.any(): return None
        delta=(a.float()-b.float()).abs()
        idx=mask.flatten().nonzero().flatten()
        return dict(shape=list(a.shape),dtype=str(a.dtype),mismatches=int(mask.sum()),numel=a.numel(),
                    max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                    first_indices=idx[:5].tolist(),a=a.flatten()[idx[:5]].tolist(),b=b.flatten()[idx[:5]].tolist())
    if a!=b:return dict(a=a,b=b)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); assert not args.output.exists(),args.output
    torch.set_num_threads(1)
    results=[]
    for rank in [0,1]:
        for file in sorted((args.a/f'rank{rank}').glob('step*.pt')):
            other=args.b/f'rank{rank}'/file.name
            if not other.exists():break
            a=torch.load(file,weights_only=False);b=torch.load(other,weights_only=False)
            af=dict(flatten(a));bf=dict(flatten(b));input_diff=[];diffs=[]
            for k,v in af.items():
                delta=compare(v,bf.get(k))
                if delta:
                    row=dict(key=k,**delta)
                    if k.startswith(('._inputs','._meta','._graph_key')):input_diff.append(row)
                    else:diffs.append(row)
            same_history=not any(x['key'].startswith('._inputs.input_ids') for x in input_diff)
            row=dict(rank=rank,step=file.stem,same_input_ids=same_history,metadata_differences=input_diff,
                     tensor_differences=len(diffs),first_output_differences=[x for x in diffs if '.out' in x['key']][:15],
                     cache_differences=[x for x in diffs if '.cache_' in x['key']],
                     target_output_differences=[x for x in diffs if x['key'].startswith('._output')])
            results.append(row)
            print(rank,file.stem,'same_input_ids',same_history,'differing_tensors',len(diffs),
                  'first_out',row['first_output_differences'][:1],flush=True)
            if not same_history:
                row['excluded']='Input IDs already differ: no causal comparison beyond this point.'
                row.pop('first_output_differences');row.pop('cache_differences');row.pop('target_output_differences')
                break
    args.output.write_text(json.dumps(dict(a=str(args.a),b=str(args.b),results=results),indent=2)+'\n')

if __name__=='__main__':main()
```
