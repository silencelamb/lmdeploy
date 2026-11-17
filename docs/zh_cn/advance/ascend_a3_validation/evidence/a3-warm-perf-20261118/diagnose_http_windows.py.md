```python
"""Decompose the same retained HTTP samples; never replace primary acceptance metrics."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def tail_speed(trace, start_token=64):
    total=0;start=None
    for ns,n in trace['chunks']:
        total+=n
        if start is None and total>=start_token:
            start=(ns,total)
        end=(ns,total)
    assert total==len(trace['token_ids'])
    return (end[1]-start[1])/((end[0]-start[0])/1e9)


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();progress=json.loads((ROOT/'results'/f'{a.label}-progress.json').read_text())
    assert progress['complete'] and progress['config']['kind']=='http'
    pairs={}
    for s in progress['sessions']:
        pairs.setdefault(s['block'],{})[s['side']]=json.loads((ROOT/'results'/f'{s["label"]}.json').read_text())
    result={'purpose':'Post-hoc diagnostic decomposition, not an acceptance replacement',
        'tail_definition':'Tokens after the first driver chunk that reaches64 cumulative tokens; retain all gaps in this window',
        'blocks':[], 'c4_request_decomposition':[], 'request_order_histogram':{}}
    hist={side:collections.Counter() for side in ['old','new']}
    first_chunks={side:collections.Counter() for side in hist}
    for block,pair in sorted(pairs.items()):
        row={'block':block,'tail_speed':{},'c4_finish_time_median':{},'decode_work':{}}
        for side,data in pair.items():
            tails={1:[],4:[]};work=collections.Counter()
            for trace in data['token_traces']:
                phase=data['phases'][str(trace['phase'])]
                if phase['kind']!='measure':continue
                tails[phase['batch']].append(tail_speed(trace))
                first_chunks[side][trace['chunks'][0][1]]+=1
            row['tail_speed'][side]={str(c):float(np.median(v)) for c,v in tails.items()}
            worker=next(w for w in data['worker_summaries'] if w['rank']==0)
            for count in worker['counts']:
                phase=data['phases'].get(str(count['phase']))
                if phase and phase['kind']=='measure' and phase['batch']==4:
                    work[str(count['signature'][1:])]+=count['count']
            row['decode_work'][side]=dict(work)
            events=[e for e in data['events'] if e['kind']=='measure' and e['batch']==4]
            row['c4_finish_time_median'][side]=float(np.median([r['events'][-1]['time'] for e in events for r in e['responses']]))
            for e in events:
                hist[side][str(sorted(range(4), key=lambda i:e['responses'][i]['ttft']))]+=1
        for old_event,new_event in zip(
            [e for e in pair['old']['events'] if e['kind']=='measure' and e['batch']==4],
            [e for e in pair['new']['events'] if e['kind']=='measure' and e['batch']==4]):
            assert old_event['rep']==new_event['rep']
            for index,(old,new) in enumerate(zip(old_event['responses'],new_event['responses'])):
                assert old['prompt']==new['prompt']
                d_first=new['ttft']-old['ttft'];d_last=new['events'][-1]['time']-old['events'][-1]['time']
                speed_change=(new['output_tps']/old['output_tps']-1)*100
                result['c4_request_decomposition'].append({'block':block,'rep':old_event['rep'],'index':index,
                    'speed_change_pct':speed_change,'first_change_ms':d_first*1000,'last_change_ms':d_last*1000,
                    'window_change_ms':(d_last-d_first)*1000,
                    'lower_stream_speed_with_earlier_first_and_last':speed_change<0 and d_first<0 and d_last<0})
        result['blocks'].append(row)
    result['request_order_histogram']={s:dict(c) for s,c in hist.items()}
    result['first_driver_chunk_token_counts']={s:dict(c) for s,c in first_chunks.items()}
    result['tail_summary']={}
    rng=np.random.default_rng(20261118);ids=rng.integers(0,len(pairs),(10000,len(pairs)))
    for c in ['1','4']:
        old=np.array([r['tail_speed']['old'][c] for r in result['blocks']]);new=np.array([r['tail_speed']['new'][c] for r in result['blocks']])
        change=(new/old-1)*100
        result['tail_summary'][c]={'old_blocks':old.tolist(),'new_blocks':new.tolist(),
            'paired_change_pct':change.tolist(),'paired_median_pct':float(np.median(change)),
            'paired_bootstrap_ci95_pct':np.quantile(np.median(change[ids],axis=1),[.025,.975]).tolist()}
    result['stream_declines_with_both_endpoints_earlier']=sum(r['lower_stream_speed_with_earlier_first_and_last'] for r in result['c4_request_decomposition'])
    result['stream_declines']=sum(r['speed_change_pct']<0 for r in result['c4_request_decomposition'])
    path=Path(a.output);assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['tail_summary','first_driver_chunk_token_counts',
        'request_order_histogram','stream_declines_with_both_endpoints_earlier','stream_declines']},indent=2))


if __name__=='__main__':main()
```
