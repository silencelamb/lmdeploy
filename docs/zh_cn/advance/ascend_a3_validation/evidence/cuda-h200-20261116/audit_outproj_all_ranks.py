import json
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
def eq(a,b):
 return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8),b.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8))
def mappings(d):
 ids={}
 for pre in d['prefills']:
  p=pre['inputs'];m=p['attn_metadata'];values=p['input_ids'].flatten()
  for i,(s,n) in enumerate(zip(m['q_start_loc'].tolist(),m['q_seqlens'].tolist())):ids[int(p['state_ids'][i])]=tuple(values[s:s+n].tolist())
 return {ids[int(sid)]:i for i,sid in enumerate(d['decode_inputs']['state_ids'].tolist())}
rows=[];name='model.language_model.layers.0.linear_attn.out_proj'
for role in ['base','pr-f64']:
 for rank in range(8):
  folder=root/'results'/f'decode-order-r2-{role}'
  a=torch.load(folder/f'orderA-rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
  b=torch.load(folder/f'orderB-rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True)
  aligned=all('prefills' in d and bool((d['model_inputs']['history_lengths']==167).all()) for d in [a,b])
  if not aligned:
   row=dict(role=role,rank=rank,aligned=False,reason='orderB is an extra decode step rather than first decode; excluded',history_lengths=[d['model_inputs']['history_lengths'].tolist() for d in [a,b]])
   rows.append(row);print(role,rank,'EXCLUDED',row['history_lengths'],flush=True);del a,b;continue
  am,bm=mappings(a),mappings(b);assert am.keys()==bm.keys()
  checks=[]
  for key in sorted(am):
   i,j=am[key],bm[key]
   checks.append(dict(left_row=i,right_row=j,input_equal=eq(a['layer_buffers'][name+'.in'][0][0,i],b['layer_buffers'][name+'.in'][0][0,j]),output_equal=eq(a['layer_buffers'][name+'.out'][0,i],b['layer_buffers'][name+'.out'][0,j])))
  rows.append(dict(role=role,rank=rank,aligned=True,checks=checks));print(role,rank,[(c['input_equal'],c['output_equal']) for c in checks],flush=True)
  del a,b
result=dict(scope='Read-only post-hoc all-rank audit of original decode-order tensors; compares orderA/orderB by identical logical prompt.',module=name,local_projection_before_reduce_not_captured=True,rows=rows)
(root/'results/outproj-all-rank-input-audit.json').write_text(json.dumps(result,indent=2))
