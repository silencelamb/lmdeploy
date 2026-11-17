"""Read-only exact BF16 cyclic-order controls for the captured NCCL partials."""
import argparse
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
def eq(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();folder=Path(a.directory)
    data=[torch.load(folder/f'rank{i}-outputs.pt',map_location='cpu',weights_only=True) for i in range(8)]
    rows=[]
    for label in ['A','B']:
        xs=[d['local'+label].reshape(-1) for d in data];target=data[0]['reduced'+label].reshape(-1)
        candidates={}
        for direction in [1,-1]:
            for start in range(8):
                order=[(start+direction*i)%8 for i in range(8)]
                value=xs[order[0]].clone()
                for rank in order[1:]:value=(value.float()+xs[rank].float()).to(torch.bfloat16)
                candidates[(direction,start)]=value
        chunks=[]
        for offset in range(0,target.numel(),2048):
            end=offset+2048;matches=[]
            for (direction,start),value in candidates.items():
                if eq(value[offset:end],target[offset:end]):matches.append(dict(direction=direction,start=start,rank_order=[(start+direction*i)%8 for i in range(8)]))
            chunks.append(dict(offset=offset,count=2048,matching_cyclic_orders=matches))
        fp64=sum(x.double() for x in xs).to(torch.bfloat16)
        row=dict(order=label,chunks=chunks,fp64_sum_matches_native=eq(fp64,target),
                 fp64_sum_matches_fp32_control=eq(fp64,data[0]['fp32reduce'+label].reshape(-1)))
        rows.append(row);print(label,[(c['offset'],[(m['direction'],m['start']) for m in c['matching_cyclic_orders']]) for c in chunks],flush=True)
    (folder/'cyclic-order-reference.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
