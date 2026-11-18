"""Compare coarse CPU18 softirq ticks overlapping actual four-request first-output windows."""
from bisect import bisect_right
import json
from pathlib import Path
from statistics import median
import hashlib
root=Path(__file__).resolve().parents[1]
snapshots=[]
p=root/'logs/host-counters-r1.jsonl'
extract=root/'results/norm-cpu18-counter-extract-r0.json'
if p.exists():
    hasher=hashlib.sha256()
    with p.open('rb') as stream:
        raw=next(stream);hasher.update(raw)
        hz=json.loads(raw)['clock_ticks']
        for line in stream:
            hasher.update(line)
            row=json.loads(line)
            cpu=next(v for v in row['stat'].splitlines() if v.startswith('cpu18 '))
            snapshots.append((row['monotonic_ns'],int(cpu.split()[7])))
    extract.write_text(json.dumps(dict(raw_sha256=hasher.hexdigest(),clock_ticks=hz,
                            fields=['monotonic_ns','cpu18_softirq_ticks'],samples=snapshots),indent=2)+'\n')
else:
    data=json.loads(extract.read_text())
    hz=data['clock_ticks']
    snapshots=data['samples']
times=[r[0] for r in snapshots]
def overlap(lo,hi):
    idx=bisect_right(times,lo)-1
    assert idx>=0 and hi<=times[-1]
    weighted=0
    while times[idx]<hi:
        t0,c0=snapshots[idx];t1,c1=snapshots[idx+1]
        assert 0<t1-t0<4e9 and c1>=c0
        fraction=(c1-c0)/hz/((t1-t0)/1e9)
        weighted+=fraction*(min(t1,hi)-max(t0,lo))
        idx+=1
    return 100*weighted/(hi-lo)
blocks=[]
for block in range(8):
    arm={}
    for mode in ('old','new'):
        data=json.loads((root/'results'/f'agents-norm-short-r0-{mode}-b{block:02d}.json').read_text())
        events=[e for e in data['events'] if e['kind']=='measure' and e['batch']==4]
        arm[mode]={}
        for e in events:
            first=sorted(t['chunks'][0][0] for t in data['token_traces'] if t['phase']==e['phase'])
            assert len(first)==4
            lo,hi=first[0],first[-1]
            arm[mode][e['rep']]=dict(start_ns=lo,end_ns=hi,spread_ms=(hi-lo)/1e6,
                                         softirq_pct=overlap(lo,hi))
    pairs=[dict(rep=i,delta_spread_ms=arm['new'][i]['spread_ms']-arm['old'][i]['spread_ms'],
                delta_softirq_pp=arm['new'][i]['softirq_pct']-arm['old'][i]['softirq_pct']) for i in range(8)]
    blocks.append(dict(block=block,samples=arm,pairs=pairs,
                  median_spread_delta_ms=median(p['delta_spread_ms'] for p in pairs),
                  median_softirq_delta_pp=median(p['delta_softirq_pp'] for p in pairs)))
out=dict(scope='CPU18 per-core softirq clock ticks sampled every 2s. Window estimates weight each overlapping interval rate by intersection length and assume a constant rate within the interval. No worker residency/IRQ handler attribution; process pairs are the independent units.',
         sample_file=str(p.relative_to(root)),clock_ticks=hz,blocks=blocks)
(root/'results/norm-cpu18-window-r0.json').write_text(json.dumps(out,indent=2)+'\n')
for b in blocks:print(b['block'],round(b['median_spread_delta_ms'],2),round(b['median_softirq_delta_pp'],2))
