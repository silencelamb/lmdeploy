"""Summarize paired same-worker prefill timings. Run from experiment root."""
import collections
import json
from pathlib import Path
from statistics import median

source = Path('results/agents-norm-isolation-r2.json')
data = json.loads(source.read_text())
assert data['complete'] and all(w['shared_weights'] and not w['dlinfer_modules'] for w in data['workers'])
samples = {}
responses = {}
for event in data['events']:
    if event['kind'] != 'diagnosis':
        continue
    key = event['batch'], event['rep'], event['mode']
    responses[key] = [r['ids'] for r in event['responses']]
    assert len(responses[key]) == event['batch']
    for worker in event['workers']:
        assert worker['mode'] == event['mode']
        row = collections.Counter()
        spans = [s for s in worker['spans'] if not s['decode']]
        row['prefill.wall'] = sum(s['host_ms'] for s in spans)
        for axis in ['wall', 'cpu']:
            for c in worker['component_calls']:
                row[c['category'] + '.' + axis] += c[axis + '_us'] / 1000
            norm = sum(c[axis + '_us'] for c in worker['norm_prefill_calls']) / 1000
            row['linear_attn.' + axis] -= norm
            row['gated_norm.' + axis] = norm
        row['other.wall'] = row['prefill.wall'] - sum(v for k,v in row.items() if k.endswith('.wall') and k != 'prefill.wall')
        assert key + (worker['rank'],) not in samples
        samples[key + (worker['rank'],)] = row
assert len(samples) == 96
summary = {}
for batch in [1,4]:
    for rep in range(12):
        assert responses[batch,rep,'old'] == responses[batch,rep,'new']
    for rank in [0,1]:
        diffs = {}
        for metric in sorted(samples[batch,0,'old',rank]):
            d = [samples[batch,i,'new',rank][metric]-samples[batch,i,'old',rank][metric] for i in range(12)]
            diffs[metric] = dict(median_ms=median(d), positive=sum(v>0 for v in d), paired_ms=d)
        summary[f'C{batch}-rank{rank}'] = diffs
Path('results/norm-component-attribution-r2.json').write_text(json.dumps(summary,indent=2)+'\n')
for label,rows in summary.items():
    print(label, {k: round(v['median_ms'],2) for k,v in rows.items()})
