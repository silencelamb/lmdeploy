# analyze_pairs.py

```python
"""Summarize all paired data, including diagnostic and GC observations."""
import argparse
import collections
import json
from pathlib import Path
import random
import statistics as st


def bootstrap_median(values):
    rng = random.Random(5134)
    medians = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(20000))
    return [medians[499], medians[19499]]


def metric(event, key):
    if key == 'throughput':
        return event['throughput']
    return st.median(row[key] for row in event['responses'])


def analyze(data):
    summary = {'complete': data['complete'], 'metrics': [], 'diagnostics': [], 'gc_stalls': []}
    for bs in (1,4):
        events = [e for e in data['events'] if e['phase']=='measure' and e['batch']==bs]
        pairs = collections.defaultdict(dict)
        for event in events:
            pairs[event['rep']][event['mode']] = event
        pairs = {key:value for key,value in pairs.items() if set(value)=={'old','new'}}
        if not pairs:
            continue
        for key in ('throughput','ttft','output_tps'):
            samples = {mode:[metric(pair[mode],key) for pair in pairs.values()] for mode in ('old','new')}
            medians = {mode:st.median(v) for mode,v in samples.items()}
            paired_pct = [(new/old-1)*100 for old,new in zip(samples['old'],samples['new'])]
            summary['metrics'].append(dict(batch=bs, metric=key, samples=samples, medians=medians,
                median_change_pct=(medians['new']/medians['old']-1)*100,
                paired_change_pct=paired_pct, paired_median_change_pct=st.median(paired_pct),
                paired_median_bootstrap_95_pct=bootstrap_median(paired_pct)))
        for mode in ('old','new'):
            events = [e for e in data['events'] if e['phase']=='diagnostic' and e['batch']==bs and e['mode']==mode]
            for rank in (0,1):
                spans = [span for event in events for worker in event['worker_after'] if worker['rank']==rank for span in worker['spans']]
                for decode in (False,True):
                    selected = [s for s in spans if s['decode']==decode]
                    if selected:
                        summary['diagnostics'].append(dict(batch=bs, mode=mode, rank=rank, decode=decode,
                            count=len(selected), captures=sum(s['capture'] for s in selected),
                            graph_count=sum(s['graph'] for s in selected), tokens=sorted({s['tokens'] for s in selected}),
                            host_median_ms=st.median(s['host_ms'] for s in selected),
                            device_median_ms=st.median(s['device_ms'] for s in selected),
                            host_max_ms=max(s['host_ms'] for s in selected),
                            device_max_ms=max(s['device_ms'] for s in selected)))
    for event in data['events']:
        if 'began' not in event:
            continue
        gc = [dict(g, pid=event['driver_gc']['pid'], role='driver') for g in event['driver_gc']['events']]
        gc += [dict(g, pid=w['gc']['pid'], role='worker') for w in event['worker_after'] for g in w['gc']['events']]
        row = event['responses'][0]
        tokens = 0
        for i,(t,n) in enumerate(row['times']):
            if i and tokens>1:
                prev = row['times'][i-1][0]
                if t-prev > .08:
                    began,ended=event['began']+prev,event['began']+t
                    overlaps=[g for g in gc if g['began']<ended and g['ended']>began]
                    summary['gc_stalls'].append(dict(batch=event['batch'],rep=event['rep'],mode=event['mode'],
                        phase=event['phase'],after_tokens=tokens,gap_ms=(t-prev)*1000,
                        began=began,ended=ended,gc=overlaps))
            tokens += n
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');a=p.parse_args()
    data=json.loads(Path(a.input).read_text());summary=analyze(data)
    output=Path(a.output) if a.output else Path(a.input).with_suffix('.summary.json')
    output.write_text(json.dumps(summary,indent=2)+'\n')
    for row in summary['metrics']:
        print(row['batch'],row['metric'],row['medians'],'median %',round(row['median_change_pct'],3),
              'paired %',round(row['paired_median_change_pct'],3),'CI',row['paired_median_bootstrap_95_pct'])
    for row in summary['diagnostics']: print('FORWARD',row)
    stalls=summary['gc_stalls'];print('GC stalls',len(stalls),'overlap',sum(bool(x['gc']) for x in stalls))
```
