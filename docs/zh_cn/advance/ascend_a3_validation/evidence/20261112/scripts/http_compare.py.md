# http_compare.py

原始验证文件快照；源码 SHA256：`3bd4dc40504d6c7f9eccbce2868e29bbb31d04ec3e2d23feebbb3a8f47805636`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import argparse
import csv
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('native',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=json.loads(a.baseline.read_text());new=json.loads(a.native.read_text())
assert old['complete'] and new['complete']
assert old['command'][1:]==new['command'][1:], 'Server configurations differ'
assert len(old['events'])==len(new['events']), 'Event counts differ'
for x,y in zip(old['events'],new['events']):
    assert (x['batch'],x['repeat'],x['warmup'])==(y['batch'],y['repeat'],y['warmup'])
    assert [r['prompt'] for r in x['responses']]==[r['prompt'] for r in y['responses']]
    assert [r['usage']['prompt_tokens'] for r in x['responses']]==[r['usage']['prompt_tokens'] for r in y['responses']]
    assert [r['usage']['completion_tokens'] for r in x['responses']]==[r['usage']['completion_tokens'] for r in y['responses']]
def quantile(xs,q):
    xs=sorted(xs);v=(len(xs)-1)*q;i=int(v);j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(v-i)
data=dict(baseline=str(a.baseline),native=str(a.native),complete=True,
    identical_server_arguments_and_workloads=True,
    metric_note='Throughput includes prefill and HTTP time. TTFT is first nonempty SSE. Output speed=(completion_tokens-1)/(last-first nonempty SSE time), a streaming proxy with MTP burst granularity.',
    functional_answers=[old['functional_answer_correct'],new['functional_answer_correct']],scenarios={})
concurrencies=sorted(set(e['batch'] for e in old['events']))
for c in concurrencies:
    values=[]
    for d in [old,new]:
        es=[e for e in d['events'] if e['batch']==c and not e['warmup']]
        rows=[r for e in es for r in e['responses']]
        values.append(dict(repetitions=len(es),requests=len(rows),
            throughput_repetitions=[e['throughput'] for e in es],
            median_throughput=statistics.median(e['throughput'] for e in es),
            ttft_p50=quantile([r['ttft'] for r in rows],.5),ttft_p95=quantile([r['ttft'] for r in rows],.95),
            output_tps_p50=quantile([r['output_tps'] for r in rows],.5),output_tps_p05=quantile([r['output_tps'] for r in rows],.05),
            slo_pass_requests=sum(r['slo_pass'] for r in rows),
            input_lengths=sorted(set(r['usage']['prompt_tokens'] for r in rows)),
            output_lengths=sorted(set(r['usage']['completion_tokens'] for r in rows))))
    before,after=values
    delta={k:100*(after[k]/before[k]-1) for k in ['median_throughput','ttft_p50','ttft_p95','output_tps_p50','output_tps_p05']}
    data['scenarios'][str(c)]=dict(baseline=before,native=after,change_percent=delta,
        paired_throughput_changes=[100*(y/x-1) for x,y in zip(before['throughput_repetitions'],after['throughput_repetitions'])],
        review_required=delta['median_throughput'] < -5 or delta['ttft_p50']>5 or delta['ttft_p95']>5 or delta['output_tps_p50'] < -5)
a.output.write_text(json.dumps(data,indent=2))
with a.output.with_suffix('.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['concurrency','environment','throughput_tps','ttft_p50_s','ttft_p95_s','output_tps_p50','SLO_pass','requests'])
    for c,row in data['scenarios'].items():
        for env in ['baseline','native']:
            d=row[env];writer.writerow([c,env,d['median_throughput'],d['ttft_p50'],d['ttft_p95'],d['output_tps_p50'],d['slo_pass_requests'],d['requests']])
print(json.dumps(data,indent=2))

```
