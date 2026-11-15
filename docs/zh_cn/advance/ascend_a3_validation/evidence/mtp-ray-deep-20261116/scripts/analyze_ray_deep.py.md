# analyze_ray_deep.py

```python
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('input',type=Path)
a=p.parse_args()
d=json.loads(a.input.read_text())
rows=[]
for r in d['records']:
    cb={c['rank']:c['ns'] for c in r['callbacks']}
    for rank,q in enumerate(r['replies']):
        i,o=q['input_stamps'],q['result_stamps']
        stamps=[r['dispatch_ns'],r['submit_end_ns'],i['deserialize_ns'],q['entered_ns'],
                q['exit_ns'],o['serialize_ns'],o['deserialize_ns'],cb[rank],r['done_ns']]
        phases=['submit','to_input_restore','restore_to_handler','handler','exit_to_serialize',
                'result_transport','restore_to_future','future_to_join']
        times={k:(y-x)/1e6 for k,x,y in zip(phases,stamps,stamps[1:])}
        rows.append(dict(item=r['item'],mode=r['mode'],rank=rank,wait_ms=r['wait_ms'],
                         delivery_ms=(q['entered_ns']-r['dispatch_ns'])/1e6,**times))
slow=[r for r in rows if max(r[k] for k in phases)>50 or r['wait_ms']>50]
result=dict(file=str(a.input),joins=len(d['records']),slow_join_count=sum(r['wait_ms']>50 for r in d['records']),
            maxima={k:max(r[k] for r in rows) for k in phases},slow=slow,
            beats=dict(driver=d['beats'],workers=[w['beats'] for w in d['workers']]),
            no_torch_npu=not any([d['torch_imported'],d['npu_imported'],
                *[v for w in d['workers'] for v in [w['torch_imported'],w['npu_imported']]]]))
out=a.input.with_name(a.input.stem+'-summary.json')
assert not out.exists(),out
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='beats'},indent=2))
```
