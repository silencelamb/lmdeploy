"""Summarize nested GDN timers without adding inclusive parents and children."""
import json
from pathlib import Path
from statistics import mean,median
from collections import defaultdict

root=Path(__file__).resolve().parents[1]
label='agents-norm-gdn-crossover-r0'
data=json.loads((root/'results'/f'{label}.json').read_text())
assert data['complete'] and len(data['workers'])==2
assert all(not w['dlinfer_modules'] and w['shared_weights'] for w in data['workers'])
samples=defaultdict(dict)
ids=defaultdict(list)
for e in data['events']:
    if e['kind']!='diagnosis':continue
    assert e['batch']==4 and data['treatments'][e['treatment']]==e['mode']
    ids[e['rep']].append([r['ids'] for r in e['responses']])
    for w in e['workers']:
        s=w['gdn_stats']
        assert s['gdn.forward'][0]==s['gdn.chunk'][0]==120
        assert len([k for k in s if k.startswith('stage.') and s[k][0]==120])==6
        assert all(t['graph'] and not t['capture'] for t in w['spans'] if t['decode'])
        pre=[t for t in w['spans'] if not t['decode']]
        assert [t['tokens'] for t in pre]==[170]*4
        s=dict(s,prefill=[4,sum(t['host_ms'] for t in pre)*1e6,sum(t['cpu_ns'] for t in w['schedule'])])
        host_reads = [v for k,v in s.items() if k.endswith('.npu') and (k.startswith('truth.') or k.startswith('method.'))]
        assert sum(v[0] for v in host_reads)==1080
        s['direct_npu_host_reads']=[sum(v[i] for v in host_reads) for i in range(3)]
        norm=w['norm_prefill_calls']
        s['gated_norm']=[len(norm),sum(t['wall_us'] for t in norm)*1000,sum(t['cpu_us'] for t in norm)*1000]
        samples[w['rank'],e['rep']][e['treatment']]=s
assert len(ids)==4 and all(len(v)==4 and all(x==v[0] for x in v) for v in ids.values())
out={}
for rank in [0,1]:
    keys=samples[rank,0][0]
    row={}
    for key in keys:
        vals=[samples[rank,rep][t][key] for rep in range(4) for t in range(4)]
        delta_wall,delta_cpu,spread=[],[],[]
        for rep in range(4):
            a,b,c,d=[samples[rank,rep][t][key] for t in range(4)]
            delta_wall.append((b[1]+d[1]-a[1]-c[1])/2e6)
            delta_cpu.append((b[2]+d[2]-a[2]-c[2])/2e6)
            spread.extend([abs(a[1]-c[1])/1e6,abs(b[1]-d[1])/1e6])
        row[key]=dict(calls_per_batch=sorted(set(v[0] for v in vals)),
                      median_wall_ms=median(v[1] for v in vals)/1e6,
                      median_cpu_ms=median(v[2] for v in vals)/1e6,
                      median_new_minus_old_wall_ms=median(delta_wall),
                      median_new_minus_old_cpu_ms=median(delta_cpu),
                      paired_wall_ms=delta_wall,paired_cpu_ms=delta_cpu,
                      median_abs_duplicate_wall_ms=median(spread))
    out[str(rank)]=row
(root/'results'/f'{label}-attribution.json').write_text(json.dumps(out,indent=2)+'\n')
for rank,row in out.items():
    print(rank, {k:(v['calls_per_batch'],round(v['median_wall_ms'],2),round(v['median_new_minus_old_wall_ms'],2),round(v['median_abs_duplicate_wall_ms'],2)) for k,v in row.items()})
