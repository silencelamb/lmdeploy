# compare_models.py

```python
"""Compare all recorded tokens and score bits, checking full key sets."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
output = []
for name in ('dense', 'sage', 'agents', 'decision', 's2preview', 's2mobius', 'qwen3next',
             'sage-confirm', 'agents-confirm', 's2preview-rdkit', 's2mobius-rdkit',
             's2preview-rdkit-all', 's2mobius-rdkit-all', 'sage-release', 'agents-release', 'agents-long'):
    paths = [ROOT/'results'/f'{name}-{side}.json' for side in ('before', 'after')]
    if name.endswith('-release'):
        paths[0] = ROOT/'results'/f'{name.removesuffix("-release")}-before.json'
    if not all(path.exists() for path in paths):
        continue
    left, right = (json.loads(path.read_text()) for path in paths)
    row = dict(model=name, complete=left['complete'] and right['complete'],
               source_files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    if not row['complete']:
        output.append(row)
        continue
    if name.endswith('-release'):
        # Compare the complete five-event trace, excluding the baseline's
        # additional benchmark events; do not alter archived source JSON.
        for item in (left, right):
            item['events'] = [e for e in item['events'] if not e['label'].startswith('bench-')]
            item['options']['suite'] = 'trace'
            item['options']['bench_concurrency'] = [1, 4]
        row['comparison_scope'] = 'all five trace events; benchmark events excluded'
    assert left['engine_config'] == right['engine_config'], name
    options = [{k: v for k, v in item['options'].items() if k != 'output'} for item in (left, right)]
    for opts in options:
        opts.setdefault('preload_rdkit', False)
    assert options[0] == options[1], name
    assert left['dlinfer_modules'] == right['dlinfer_modules'] == []
    assert len(left['events']) == len(right['events'])
    mismatches=[]
    responses=tokens=scores=0
    for a, b in zip(left['events'], right['events']):
        assert a['label'] == b['label'] and a['prompts'] == b['prompts']
        assert len(a['responses']) == len(b['responses'])
        for idx, (x, y) in enumerate(zip(a['responses'], b['responses'])):
            responses += 1
            tokens += len(x['token_ids'])
            where=f'{a["label"]}:{idx}'
            if x['token_ids'] != y['token_ids']:
                mismatches.append(dict(where=where, reason='token_ids'))
            for key in ('input_token_len', 'finish_reason', 'cached_tokens'):
                assert x[key] == y[key], (where, key)
            xs, ys = x.get('logprobs') or [], y.get('logprobs') or []
            assert len(xs) == len(ys), where
            for step, (xt, yt) in enumerate(zip(xs, ys)):
                if set(xt) != set(yt):
                    mismatches.append(dict(where=where, step=step, reason='topk_token_set'))
                    continue
                for token in xt:
                    xv, yv = xt[token], yt[token]
                    assert math.isfinite(xv) and math.isfinite(yv)
                    scores += 1
                    if struct.pack('!f', xv) != struct.pack('!f', yv) or struct.pack('!d', xv) != struct.pack('!d', yv):
                        mismatches.append(dict(where=where, step=step, token=token, reason='score_bits',
                                               before=xv, after=yv))
    perf=[]
    for batch in (1, 4):
        events=[[event for event in item['events'] if event.get('batch')==batch and not event.get('warmup')]
                for item in (left,right)]
        if not all(events):
            continue
        metrics=[]
        for group in events:
            metrics.append(dict(throughput=statistics.median(e['throughput'] for e in group),
                                ttft=statistics.median(statistics.median(r['ttft'] for r in e['responses']) for e in group),
                                decode_tps=statistics.median(statistics.median(r['output_tps'] for r in e['responses']) for e in group)))
        a,b=metrics
        perf.append(dict(batch=batch, before=a, after=b,
                         throughput_change_pct=(b['throughput']/a['throughput']-1)*100,
                         ttft_change_pct=(b['ttft']/a['ttft']-1)*100,
                         needs_confirmation=b['throughput']/a['throughput']<.95 or b['ttft']/a['ttft']>1.1))
    row.update(responses=responses, generated_tokens=tokens, recorded_scores=scores,
               mismatch_count=len(mismatches), first_mismatches=mismatches[:10], bitwise_equal=not mismatches,
               performance=perf)
    output.append(row)
(ROOT/'results/model-comparison.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
```
