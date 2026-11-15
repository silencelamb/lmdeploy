# gdn_oracle.py

```python
"""Independent CPU FP32 recurrence, including every circular state write.

Predeclared state tolerance: atol=1e-5, rtol=1e-4. BF16 output tolerance:
atol=1e-3, rtol=1e-2. Exact repeated-state equality is checked separately.
"""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('files',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists(),a.output
torch.set_num_threads(1)
rows=[]
for path in a.files:
    d=torch.load(path,weights_only=False)
    q=d['q'].float();k=d['k'].float();v=d['v'].float();g=d['g'].float();beta=d['beta'].float()
    _,T,H,K=q.shape;HV=v.shape[2]
    q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
    q=q.repeat_interleave(HV//H,dim=2);k=k.repeat_interleave(HV//H,dim=2)
    state=d['initial'].clone();h=state[7].clone();outs=[]
    for t in range(T):
        h=h*torch.exp(g[0,t])[:,None,None]
        err=v[0,t]-(h*k[0,t][:,None,:]).sum(-1)
        err=err*beta[0,t][:,None]
        h=h+err[:,:,None]*k[0,t][:,None,:]
        outs.append((h*(q[0,t]/K**.5)[:,None,:]).sum(-1))
        state[4+(27+t+1)%4]=h
    out=torch.stack(outs,dim=0).unsqueeze(0)
    row=dict(file=str(path),state_max_abs=float((state-d['state']).abs().max()),
        output_max_abs=float((out-d['output'].float()).abs().max()),
        state_close=torch.allclose(state,d['state'],atol=1e-5,rtol=1e-4),
        output_close=torch.allclose(out,d['output'].float(),atol=1e-3,rtol=1e-2))
    bad=path.with_name(path.stem+'-first-bad.pt')
    if bad.exists():
        b=torch.load(bad,weights_only=False)
        row['bad_state_max_abs']=float((state-b['state']).abs().max())
        row['bad_state_close']=torch.allclose(state,b['state'],atol=1e-5,rtol=1e-4)
    rows.append(row);print(row)
a.output.write_text(json.dumps(dict(state_atol=1e-5,state_rtol=1e-4,output_atol=1e-3,output_rtol=1e-2,results=rows),indent=2)+'\n')
```
