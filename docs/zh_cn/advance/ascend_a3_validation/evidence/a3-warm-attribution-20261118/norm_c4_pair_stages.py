"""Compare client-visible phases of every matched C4 batch in the formal norm64 run."""
from collections import Counter
import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[1]
def segments(event):
    timestamps=sorted(r['times'][0][0] for r in event['responses'])
    last_first=timestamps[-1]
    received=sum(n for r in event['responses'] for t,n in r['times'] if t<=last_first)
    assert received==4
    final=max(r['times'][-1][0] for r in event['responses'])
    edges=[0,*timestamps,final,event['elapsed']]
    return [(b-a)*1000 for a,b in zip(edges,edges[1:])]
blocks=[]
hist=Counter()
for block in range(8):
    groups=[]
    for mode in ('old','new'):
        p=root/'results'/f'agents-norm-short-r0-{mode}-b{block:02d}.json'
        data=json.loads(p.read_text())
        group={e['rep']:e for e in data['events'] if e['kind']=='measure' and e['batch']==4}
        assert len(group)==8
        for e in group.values():
            for w in data['worker_summaries']:
                pre=[c for c in w['counts'] if c['phase']==e['phase'] and not c['signature'][1]]
                assert len(pre)==1 and pre[0]['signature'][2:]==[170,1] and pre[0]['count']==4
        groups.append(group)
    diffs=[]
    for rep in range(8):
        old,new=[g[rep] for g in groups]
        assert [r['token_ids'] for r in old['responses']]==[r['token_ids'] for r in new['responses']]
        a,b=segments(old),segments(new)
        diffs.append([y-x for x,y in zip(a,b)])
    blocks.append(dict(block=block,pairs_ms=diffs,median_spread_delta_ms=median(sum(x[1:4]) for x in diffs),
                       median_stage_delta_ms=[median(x) for x in zip(*diffs)]))
out=dict(scope='8 blocks, 8 individually matched C4 batches per block. All 128 old/new batches had four singleton 170-token prefill forwards on both ranks. Exactly four output tokens had arrived when the last request received its first token in all 128 batches.',stages=['first','first to second','second to third','third to fourth','fourth first to final','post-final'],blocks=blocks)
(root/'results'/'norm-c4-paired-stages-r0.json').write_text(json.dumps(out,indent=2)+'\n')
print('matched spread', [round(b['median_spread_delta_ms'],1) for b in blocks], 'overall',median(b['median_spread_delta_ms'] for b in blocks))
print('stage medians', [round(median(z),2) for z in zip(*(b['median_stage_delta_ms'] for b in blocks))])
