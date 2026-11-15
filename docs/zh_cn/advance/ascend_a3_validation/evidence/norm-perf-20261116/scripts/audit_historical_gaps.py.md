# audit_historical_gaps.py

```python
"""Describe old streaming gaps without modifying/excluding raw samples."""
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
old=Path('/root/workspace/a3-gated-norm-20261115/results')
records=[]
for side in ('before','after'):
    path=old/f'agents-long-{side}.json'
    data=json.loads(path.read_text())
    for event in data['events']:
        if event['warmup']:continue
        row=event['responses'][0]
        tokens=0;gaps=[]
        for i,(t,n) in enumerate(row['times']):
            if i and t-row['times'][i-1][0]>.08:
                gaps.append(dict(after_tokens=tokens,duration_ms=(t-row['times'][i-1][0])*1000,
                                 began=row['times'][i-1][0],ended=t))
            tokens+=n
        records.append(dict(side=side,label=event['label'],source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed=event['elapsed'],
            throughput=event['throughput'],ttft=[r['ttft'] for r in event['responses']],
            median_decode_tps=st.median(r['output_tps'] for r in event['responses']),
            row0_long_gaps=gaps,post_first_token_long_gaps_ms=sum(g['duration_ms'] for g in gaps if g['after_tokens']>1)))
(ROOT/'results/historical-stream-gaps.json').write_text(json.dumps(records,indent=2)+'\n')
print('wrote',len(records),'original measured batches; no samples removed')
```
