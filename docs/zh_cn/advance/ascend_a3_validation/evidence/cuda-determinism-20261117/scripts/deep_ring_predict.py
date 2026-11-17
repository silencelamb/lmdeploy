"""Predict actual NCCL RING_LL output from captured BF16 partials and source order."""
import argparse
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
def same(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def accumulate(parts,order):
    y=parts[order[0]].clone()
    for r in order[1:]:y=(y.float()+parts[r].float()).to(torch.bfloat16)
    return y
def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();folder=Path(a.directory)
    d=[torch.load(folder/f'rank{r}-outputs.pt',map_location='cpu',weights_only=True) for r in range(8)]
    rows=[]
    for label in ['A','B']:
        parts=[x['local'+label].flatten() for x in d];observed=d[0]['reduced'+label].flatten()
        predicted=torch.empty_like(observed)
        for chunk in range(8):
            lo=chunk*2048;hi=lo+2048;order=[(chunk+1+j)%8 for j in range(8)]
            predicted[lo:hi]=accumulate([x[lo:hi] for x in parts],order)
        rows.append(dict(order=label,all_elements_equal=same(predicted,observed),elements=observed.numel()))
    perm=json.load(open(folder/'rank0.json'))['permutation'];example=None
    for b_row,a_row in enumerate(perm):
        aa=d[0]['reducedA'][0,a_row];bb=d[0]['reducedB'][0,b_row]
        if not same(aa,bb):
            col=int((aa!=bb).nonzero()[0]);parts=[float(x['localA'][0,a_row,col]) for x in d]
            ac=(a_row*4096+col)//2048;bc=(b_row*4096+col)//2048
            ao=[(ac+1+i)%8 for i in range(8)];bo=[(bc+1+i)%8 for i in range(8)]
            def trace(order):
                out=[];value=torch.tensor(parts[order[0]],dtype=torch.bfloat16);out.append(float(value))
                for rank in order[1:]:value=(value.float()+torch.tensor(parts[rank])).to(torch.bfloat16);out.append(float(value))
                return out
            example=dict(old_row=a_row,new_row=b_row,column=col,unchanged_rank_partials=parts,rank_order_A=ao,rank_order_B=bo,
                         rounded_partial_sums_A=trace(ao),rounded_partial_sums_B=trace(bo),actual_A=float(aa[col]),actual_B=float(bb[col]),fp64_sum=sum(parts))
            break
    result=dict(assumptions='Observed one-channel RING_LL, 16384 BF16 elements,8ranks; 2048-element chunks; ring0->1->...->7; NCCL runRing reduces chunk c in cyclic order c+1,...,c.',
                source_url='https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/all_reduce.h',checks=rows,example=example)
    (folder/'ring-source-prediction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
