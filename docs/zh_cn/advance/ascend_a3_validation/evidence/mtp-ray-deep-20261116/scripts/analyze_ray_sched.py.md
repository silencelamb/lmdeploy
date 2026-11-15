# analyze_ray_sched.py

```python
import bisect
import json
from pathlib import Path
import sys

src=Path(sys.argv[1]);d=json.loads(src.read_text());root=src.parent.parent/'artifacts'/src.stem
samples=[json.loads(l) for l in (root/'sched.jsonl').read_text().splitlines()]
ns=[r['ns'] for r in samples]
results=[]
for r in d['records']:
    if r['wait_ms']<=50:continue
    i=max(0,bisect.bisect_right(ns,r['dispatch_ns'])-1)
    j=min(len(ns)-1,bisect.bisect_left(ns,r['done_ns'])+1)
    tasks={}
    for tid,x in samples[j]['tasks'].items():
        y=samples[i]['tasks'].get(tid)
        if not y:continue
        states=[s['tasks'].get(tid,{}) for s in samples[i:j+1]]
        tasks[tid]=dict(runqueue_ms=(x['runqueue_ns']-y['runqueue_ns'])/1e6,
            runtime_ms=(x['runtime_ns']-y['runtime_ns'])/1e6,nice=x['nice'],
            states={v:sum(t.get('state')==v for t in states) for v in ['R','S','D']},
            cpus=sorted(set(t['processor'] for t in states if 'processor' in t)))
    results.append(dict(item=r['item'],mode=r['mode'],wait_ms=r['wait_ms'],
        delivery=[dict(pid=q['pid'],tid=q['tid'],ms=(q['entered_ns']-r['dispatch_ns'])/1e6) for q in r['replies']],
        tasks=tasks))
out=src.with_name(src.stem+'-sched-summary.json');assert not out.exists(),out
out.write_text(json.dumps(results,indent=2)+'\n')
for r in results:
    print(r['item'],r['wait_ms'],r['delivery'])
    print({tid:x for tid,x in r['tasks'].items() if x['runqueue_ms']>50 or x['runtime_ms']>50 or x['states']['R']>5 or x['states']['D']>5})
```
