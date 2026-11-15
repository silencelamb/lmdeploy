# summarize_deep.py

```python
"""Summaries from preserved artifacts; do not overwrite first-round evidence."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/'results'/n).read_text())
out=ROOT/'results/deep-investigation-summary-r2.json'
assert not out.exists(),out
result=dict(native_base='5f1b1b3c12b1c198813f65275f778c7ad292fb02',
            fix_commit='dfa3049f1575305b7564d6f15c48022661bd902a')
aa,bb=(read(f'mtp-stage1-suite-r{i}.json') for i in (0,1))
matches=[]
for ea,eb in zip(aa['events'],bb['events']):
    for i,(a,b) in enumerate(zip(ea['responses'],eb['responses'])):
        matches.append(dict(case=ea['label'],request=i,token_ids_equal=a['token_ids']==b['token_ids'],
            top5_logprobs_equal=a['logprobs']==b['logprobs'],tokens=len(a['token_ids'])))
audits=[]
for i in (0,1):
    p=ROOT/f'results/mtp-runtime-stage1-suite-r{i}-rank0.jsonl'
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    audits.append(dict(run=i,verifier_calls=len(rows),request_rows=sum(r['shape'][0] for r in rows),
        all_greedy=all(r['max_top_k']==1 for r in rows),all_cpu_oracle_equal=all(r['cpu_oracle_equal'] for r in rows),
        rng_unchanged=all(r['cpu_rng_unchanged'] and r['npu_rng_unchanged'] for r in rows)))
result['mtp_suite']=dict(requests=matches,audits=audits)
result['gdn']={}
for n in ['gdn-fixed-eager-r0','gdn-fixed-graph-r0','gdn-stage1-eager-r0','gdn-stage1-graph-r0',
          'gdn-real-base-graph-r1','gdn-real-fixed-graph-r0','gdn-timing-original-r0','gdn-timing-fixed-r0']:
    d=read(n+'.json');r=d['results']
    row=dict(iterations=len(r),differing_outputs=sum(x['output_mismatch']>0 for x in r),
        differing_states=sum(x['state_mismatch']>0 for x in r),max_state_delta=max(x['state_max_abs'] for x in r),
        input_hashes=d['input_hashes'])
    if 'device_ms' in r[0]:row['device_median_ms']=statistics.median(x['device_ms'] for x in r[10:])
    result['gdn'][n]=row
result['ray']={}
for n in ['ray-deep-default-r0','ray-deep-poll-r0','ray-deep-nostats-r0','ray-deep-noacc-strace-r0',
          'ray-noacc-nice0-r0','ray-noacc-nojemalloc-r0','ray-noacc-threads1-r0','ray-sched-default-r0',
          'ray-sched-all-r0','ray-affinity-node1-r0','ray-exclude18-r0','ray-exclude18-r1','ray-exclude18-gc-r0','ray-exclude18-gcoff-r0']:
    d=read(n+'.json');rs=d['records']
    result['ray'][n]=dict(joins=len(rs),over50=sum(x['wait_ms']>50 for x in rs),
        over300=sum(x['wait_ms']>300 for x in rs),max_wait_ms=max(x['wait_ms'] for x in rs),
        max_delivery_ms=max((q['entered_ns']-r['dispatch_ns'])/1e6 for r in rs for q in r['replies']),
        torch_imported=d['torch_imported'],npu_imported=d['npu_imported'],acl_imported=d.get('acl_imported'),
        diagnostic_ptrace='strace' in n or n=='ray-deep-poll-r0')
result['pipe']={}
for n in ['pipe-nice15-r0','pipe-cpu18-r0','pipe-cpu18-r1','pipe-cpu18-r2','pipe-cpu42-r0']:
    d=read(n+'.json');rs=d['records'];sleeps=[(r['join_ns']-r['sent_ns'])/1e6 for r in rs]
    starts=[r['sent_ns'] for r,t in zip(rs,sleeps) if t>50]
    periods=[(y-x)/1e9 for x,y in zip(starts,starts[1:])]
    result['pipe'][n]=dict(iterations=len(rs),sleep_over50=sum(t>50 for t in sleeps),
        max_sleep_ms=max(sleeps),median_stall_period_s=statistics.median(periods) if periods else None)
http=read('http-exclude18-r0-summary.json')
result['http']=dict(complete=http['http_complete'],requests=http['measured_requests'],rpc_joins=http['rpc_count'],
    rpc_over50=len(http['long_rpc_waits']),steady_sse_over100=sum(not r['warmup'] and r['chunk_index']>8 for r in http['sse_gaps_over_100ms']))
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
```
