```python
"""Cluster-paired statistics; confidence intervals must establish the claimed scope."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = {'throughput': 'higher', 'stream_tps_p50': 'higher', 'ttft_p50': 'lower',
           'ttft_p95': 'lower', 'tpot_p50': 'lower'}


def metrics(data, concurrency):
    events = [e for e in data['events'] if e['kind']=='measure' and e['batch']==concurrency]
    rows = [r for e in events for r in e['responses']]
    ttft = np.array([r['ttft'] for r in rows], dtype=float)
    speed = np.array([r['output_tps'] for r in rows], dtype=float)
    return {'throughput': float(np.median([e['throughput'] for e in events])),
            'stream_tps_p50': float(np.median(speed)), 'ttft_p50': float(np.median(ttft)),
            'ttft_p95': float(np.quantile(ttft,.95)), 'tpot_p50': float(np.median(1/speed))}, ttft


def classify(interval, direction):
    lo,hi = interval
    if lo >= -5 and hi <= 5:
        return 'equivalent'
    if (direction=='higher' and lo>0) or (direction=='lower' and hi<0):
        return 'improved'
    if (direction=='higher' and hi < -5) or (direction=='lower' and lo>5):
        return 'regression'
    return 'inconclusive'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    progress_path=ROOT/'results'/f'{a.label}-progress.json'
    progress=json.loads(progress_path.read_text())
    assert progress['complete'] and len(progress['blocks'])==progress['planned_pairs']>=8
    assert all(x['exact_token_work'] for x in progress['blocks'])
    pairs={}
    sources={}
    for s in progress['sessions']:
        path=ROOT/'results'/f'{s["label"]}.json'
        raw=path.read_bytes();data=json.loads(raw)
        assert data['complete'] and data['warmup_qualified'] and data['measurement_qualified']
        pairs.setdefault(s['block'],{})[s['side']]=data
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
    rng=np.random.default_rng(20261118)
    ids=rng.integers(0,len(pairs),(10000,len(pairs)))
    result={'comparison':progress['comparison'],'config':progress['config'],
            'pairs':len(pairs),'bootstrap_seed':20261118,'bootstrap_draws':10000,
            'acceptance_band_pct':[-5,5],'statistical_unit':'fresh-process paired block',
            'source_sha256':sources,'metrics':{}}
    for c in [1,4]:
        summaries={s:[] for s in ['old','new']};latencies={s:[] for s in summaries}
        for block,pair in sorted(pairs.items()):
            for side in summaries:
                m,t=metrics(pair[side],c);summaries[side].append(m);latencies[side].append(t)
        output={}
        for name,direction in METRICS.items():
            old=np.array([x[name] for x in summaries['old']]);new=np.array([x[name] for x in summaries['new']])
            delta=(new/old-1)*100
            distribution=np.median(delta[ids],axis=1)
            ci=np.quantile(distribution,[.025,.975]).tolist()
            row={'old_blocks':old.tolist(),'new_blocks':new.tolist(),
                 'paired_change_pct':delta.tolist(),'paired_median_pct':float(np.median(delta)),
                 'paired_ci95_pct':ci,'ratio_of_block_medians_pct':float((np.median(new)/np.median(old)-1)*100),
                 'status':classify(ci,direction),'direction':direction}
            if name=='ttft_p95':
                old_t=np.stack(latencies['old']);new_t=np.stack(latencies['new'])
                old_q=np.quantile(old_t[ids].reshape(len(ids),-1),.95,axis=1)
                new_q=np.quantile(new_t[ids].reshape(len(ids),-1),.95,axis=1)
                global_ci=np.quantile((new_q/old_q-1)*100,[.025,.975]).tolist()
                row.update(global_old_p95=float(np.quantile(old_t,.95)),
                           global_new_p95=float(np.quantile(new_t,.95)),
                           global_p95_ratio_ci95_pct=global_ci,global_status=classify(global_ci,direction))
                if row['status'] in ['equivalent','improved'] and row['global_status'] not in ['equivalent','improved']:
                    row['status']='inconclusive'
            output[name]=row
        result['metrics'][str(c)]=output
    result['all_metrics_accepted']=all(v['status'] in ['equivalent','improved'] for group in result['metrics'].values() for v in group.values())
    out=Path(a.output);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    for c,group in result['metrics'].items():
        for name,value in group.items():print(c,name,round(value['paired_median_pct'],3),value['paired_ci95_pct'],value['status'])


if __name__=='__main__':main()
```
