# analyze_stages.py

```python
"""Match client stalls to timed Ray submission/previous-RPC waits."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
records=[]
for name in ('agents-stage-profile','agents-await-profile'):
    path=ROOT/'results'/f'{name}.json'
    data=json.loads(path.read_text())
    for event in data['events']:
        row=event['responses'][0]
        gaps=[];tokens=0
        for i,(t,n) in enumerate(row['times']):
            if i and tokens>1 and t-row['times'][i-1][0]>.08:
                gaps.append(dict(after_tokens=tokens,began=event['began']+row['times'][i-1][0],
                                 ended=event['began']+t,duration_ms=(t-row['times'][i-1][0])*1000))
            tokens+=n
        for stage in event['driver']['slow_stages']:
            if stage['name'] not in ('driver.previous_forward_rpc','driver.submit_inputs'):
                continue
            overlaps=[g for g in gaps if stage['began']<g['ended'] and stage['ended']>g['began']]
            records.append(dict(run=name,batch=event['batch'],rep=event['rep'],stage=stage,
                overlapping_stream_gaps=overlaps,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
repo='/root/workspace/a3-native-pr-20261115/lmdeploy'
unchanged={}
for file in ('lmdeploy/pytorch/engine/executor/ray_executor.py','lmdeploy/pytorch/backends/ascend/graph_runner.py'):
    blobs={sha:subprocess.check_output(['git','rev-parse',sha+':'+file],cwd=repo,text=True).strip()
           for sha in ('342db85ecff3c0dc864a4bf8613c589c270f4141','5f1b1b3c12b1c198813f65275f778c7ad292fb02')}
    unchanged[file]=dict(blobs=blobs,identical=len(set(blobs.values()))==1)
result=dict(matches=records,source_audit=unchanged,
    interpretation='Measured stalls localize to awaiting previous worker submission RPCs, not proof of the Ray-internal trigger or every historical stall.')
(ROOT/'results/ray-wait-localization.json').write_text(json.dumps(result,indent=2)+'\n')
for r in records:
    s=r['stage'];print(r['run'],r['batch'],r['rep'],s['name'],round(s['wall_ms'],3),
                      'cpu_ms',round(s['thread_cpu_ms'],3),'gaps',[(g['after_tokens'],round(g['duration_ms'],3)) for g in r['overlapping_stream_gaps']])
print(unchanged)
```
