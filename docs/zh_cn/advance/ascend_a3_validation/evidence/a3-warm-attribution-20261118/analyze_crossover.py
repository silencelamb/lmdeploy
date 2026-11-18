"""Analyze diagnostic crossover; all times are ms per whole prefill batch."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys

label = sys.argv[1]
root = Path(__file__).resolve().parents[1]
exp = json.loads((root/'results'/f'{label}.json').read_text())
assert exp['complete'] and len(exp['workers']) == 2
assert all(w['shared_weights'] and not w['dlinfer_modules'] for w in exp['workers'])
groups = defaultdict(dict)
tokens = defaultdict(list)
orders = defaultdict(list)
for event in exp['events']:
    if event['kind'] != 'diagnosis':
        continue
    bs, rep, arm = event['batch'], event['rep'], event['treatment']
    assert event['mode'] == exp['treatments'][arm]
    tokens[bs,rep].append([r['ids'] for r in event['responses']])
    orders[bs,rep].append(arm)
    assert all(r['cached_tokens'] == 0 and len(r['ids']) == 64 for r in event['responses'])
    for w in event['workers']:
        row = Counter()
        key = bs, rep, w['rank']
        assert arm not in groups[key]
        pre = [s for s in w['spans'] if not s['decode']]
        dec = [s for s in w['spans'] if s['decode']]
        assert len(pre)==bs and all(s['tokens']==170 for s in pre)
        assert dec and all(s['graph'] and not s['capture'] for s in dec)
        assert len(w['schedule'])==bs
        row['prefill.wall'] = sum(s['host_ms'] for s in pre)
        for s in w['schedule']:
            row['prefill.cpu'] += s['cpu_ns']/1e6
            row['prefill.runqueue'] += s['runqueue_ns']/1e6
            row['prefill.offcpu'] += (s['wall_ns']-s['cpu_ns'])/1e6
        for timer in w['component_calls']:
            if timer['category']=='linear_attn':
                row['linear_attn.wall']+=timer['wall_us']/1000
                row['linear_attn.cpu']+=timer['cpu_us']/1000
        for timer in w['norm_prefill_calls']:
            row['norm.wall']+=timer['wall_us']/1000
            row['norm.cpu']+=timer['cpu_us']/1000
        counts = Counter()
        for name,wall,cpu in w['linear_detail']:
            cat = name.rsplit('.',1)[1]
            counts[cat]+=1
            row[cat+'.wall']+=wall/1e6
            row[cat+'.cpu']+=cpu/1e6
        assert set(counts)=={'in_proj_qkv','in_proj_z','in_proj_ba','conv1d','out_proj'}
        assert set(counts.values())=={30*bs},counts
        for axis in ('wall','cpu'):
            row['linear_other.'+axis] = row['linear_attn.'+axis]-row['norm.'+axis]-sum(row[c+'.'+axis] for c in counts)
        for s in w['schedule']:
            row['switch.voluntary'] += s['voluntary']
            row['switch.involuntary'] += s['involuntary']
        groups[key][arm] = row

assert len(groups)==32 and len(tokens)==16
for (bs,rep),calls in tokens.items():
    assert len(calls)==4 and all(call==calls[0] for call in calls)
    assert orders[bs,rep]==exp['orders'][rep%4]
summary = {}
for bs in (1,4):
    for rank in (0,1):
        rows = [groups[bs,rep,rank] for rep in range(8)]
        result={}
        for metric in rows[0][0]:
            contrasts = []
            duplicate = []
            for row in rows:
                a,b,c,d = [row[i][metric] for i in range(4)]
                contrasts.append(mean([b,d])-mean([a,c]))
                duplicate.append([c-a,d-b])
            result[metric] = dict(paired_ms=contrasts, paired_median_ms=median(contrasts),
                positive=sum(v>0 for v in contrasts), same_mode_ms=duplicate,
                median_abs_duplicate_ms=median(abs(v) for pair in duplicate for v in pair))
        summary[f'C{bs}-rank{rank}'] = result
dest=root/'results'/f'{label}-attribution.json'
dest.write_text(json.dumps(summary,indent=2)+'\n')
for name,metrics in summary.items():
    print(name, {k:(round(v['paired_median_ms'],2),round(v['median_abs_duplicate_ms'],2)) for k,v in metrics.items()})
