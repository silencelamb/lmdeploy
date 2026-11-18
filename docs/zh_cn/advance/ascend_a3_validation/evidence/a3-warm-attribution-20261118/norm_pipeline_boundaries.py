"""Exact additive decomposition of block median elapsed time at stream boundaries."""
import json
from pathlib import Path
from statistics import mean,median

root=Path(__file__).resolve().parents[1]
rows={}
for batch in (1,4):
    result=[]
    for block in range(8):
        modes={}
        for mode in ('old','new'):
            path=root/'results'/f'agents-norm-short-r0-{mode}-b{block:02d}.json'
            data=json.loads(path.read_text())
            events=sorted((e for e in data['events'] if e['kind']=='measure' and e['batch']==batch),key=lambda e:e['elapsed'])
            assert len(events)==(25 if batch==1 else 8)
            middle=events[(len(events)-1)//2:len(events)//2+1]
            parts=[]
            for e in middle:
                first=[r['times'][0][0] for r in e['responses']]
                last=max(r['times'][-1][0] for r in e['responses'])
                edges=(0,min(first),max(first),last,e['elapsed'])
                part=[(b-a)*1000 for a,b in zip(edges,edges[1:])]
                assert all(v>=0 for v in part)
                parts.append(part)
            t=[mean(v) for v in zip(*parts)]
            assert abs(sum(t)-1000*median(e['elapsed'] for e in events))<1e-6
            modes[mode]=dict(reps=[e['rep'] for e in middle],stages_ms=t)
        delta=[n-o for n,o in zip(modes['new']['stages_ms'],modes['old']['stages_ms'])]
        result.append(dict(block=block,modes=modes,delta_ms=delta,total=sum(delta)))
    rows[f'C{batch}']=result
out=dict(scope='Client-side streaming boundaries of the median elapsed batch (mean of middle two for even sample). Diagnostic; overlap includes mixed prefill/decode/scheduling/client work.',stages=['first arrival','first-arrival spread','last first arrival to final arrival','post-final cleanup'],blocks=rows)
(root/'results'/'norm-pipeline-boundaries-r0.json').write_text(json.dumps(out,indent=2)+'\n')
for batch,records in rows.items():
    print(batch,'median total',median(r['total'] for r in records),'per-stage (median,mean,positive)',[(round(median(v),2),round(mean(v),2),sum(i>0 for i in v)) for v in zip(*(r['delta_ms'] for r in records))])
