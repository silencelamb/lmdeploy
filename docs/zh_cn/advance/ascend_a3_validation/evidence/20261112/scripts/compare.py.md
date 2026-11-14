# compare.py

原始验证文件快照；源码 SHA256：`c7f95c337cff420032deab6d72e4873d5679176a3e857a44fb11529b72fbd5c3`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Compare saved inference traces without hiding mismatches after token divergence."""
import argparse
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('baseline',type=Path)
p.add_argument('native',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
old=json.loads(a.baseline.read_text());new=json.loads(a.native.read_text())
result=dict(baseline=str(a.baseline),native=str(a.native),complete=old['complete'] and new['complete'],
            exact_sequences=0,sequences=0,tokens=0,first_mismatches=[],score_diffs=[],bench={})
assert old['engine_config']==new['engine_config'],'Configurations differ'
for option in ['model','mtp','new_tokens','suite','length','no_thinking','respect_eos','skip_input_scoring']:
    assert old['options'].get(option)==new['options'].get(option),f'Options differ: {option}'
assert len(old['events'])==len(new['events']),'Event counts differ'
score_diffs=[]
fixed_diffs=[]
fixed_rows=0
fixed_top1_mismatches=[]
for e,f in zip(old['events'],new['events']):
    assert e['label']==f['label']
    assert e.get('prompts')==f.get('prompts')
    assert len(e['responses'])==len(f['responses'])
    if e['label'].startswith('bench-'):
        continue
    for i,(x,y) in enumerate(zip(e['responses'],f['responses'])):
        result['sequences']+=1
        result['tokens']+=len(x['token_ids'])
        exact=x['token_ids']==y['token_ids']
        result['exact_sequences']+=int(exact)
        if e['label']=='fixed-history-scores':
            assert len(x['logprobs'])==len(y['logprobs'])
            for j,(xs,ys) in enumerate(zip(x['logprobs'],y['logprobs'])):
                fixed_rows+=1
                fixed_diffs.extend(abs(xs[token]-ys[token]) for token in xs.keys() & ys.keys())
                if max(xs,key=xs.get)!=max(ys,key=ys.get):
                    fixed_top1_mismatches.append(dict(request=i,position=j,baseline=xs,native=ys))
        for j,(u,v) in enumerate(zip(x['token_ids'],y['token_ids'])):
            # This position has the same preceding token history, including the
            # first differing generated token. Later positions are incomparable.
            xs=(x.get('logprobs') or [])[j] if x.get('logprobs') else {}
            ys=(y.get('logprobs') or [])[j] if y.get('logprobs') else {}
            for token in xs.keys() & ys.keys():
                score_diffs.append(abs(xs[token]-ys[token]))
            if u!=v:
                result['first_mismatches'].append(dict(event=e['label'],request=i,position=j,
                    baseline_token=u,native_token=v,baseline_scores=xs,native_scores=ys))
                break
result['numerics']=dict(max_abs=max(score_diffs,default=0),
    rms=math.sqrt(sum(x*x for x in score_diffs)/max(1,len(score_diffs))),compared_scores=len(score_diffs))
result.pop('score_diffs')
result['token_pass']=result['complete'] and result['sequences']>0 and result['sequences']==result['exact_sequences']
result['numerical_pass']=result['complete'] and len(score_diffs)>0 and result['numerics']['max_abs']<=.02 and result['numerics']['rms']<=.005
result['numerical_status']='unverified-no-scores' if not score_diffs else ('pass' if result['numerical_pass'] else 'fail')
if fixed_rows:
    result['fixed_history']=dict(rows=fixed_rows,compared_scores=len(fixed_diffs),
        max_abs=max(fixed_diffs),rms=math.sqrt(sum(x*x for x in fixed_diffs)/len(fixed_diffs)),
        top1_mismatches=fixed_top1_mismatches)
    result['fixed_history']['pass']=(max(fixed_diffs)<=.02 and
        result['fixed_history']['rms']<=.005 and not fixed_top1_mismatches)
def quantile(xs,q):
    xs=sorted(xs); index=(len(xs)-1)*q; lo=int(index); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(index-lo)
for bs in [1,4,16]:
    summaries=[]
    for data in [old,new]:
        events=[e for e in data['events'] if e.get('batch')==bs and not e.get('warmup',True)]
        if not events: break
        rows=[r for e in events for r in e['responses']]
        throughput=[e['throughput'] for e in events]
        intervals=[(r['times'][j][0]-r['times'][j-1][0])/r['times'][j][1]
                   for r in rows for j in range(1,len(r['times']))]
        summaries.append(dict(repetitions=len(events),requests=len(rows),throughput=throughput,
            median_throughput=statistics.median(throughput),
            ttft_p50=quantile([r['ttft'] for r in rows],.5),ttft_p95=quantile([r['ttft'] for r in rows],.95),
            output_tps_p50=quantile([r['output_tps'] for r in rows],.5),
            amortized_itl_p50=quantile(intervals,.5),amortized_itl_p95=quantile(intervals,.95),
            slo_pass_requests=sum(r['slo_pass'] for r in rows)))
    if len(summaries)==2:
        x,y=summaries
        change={k:100*(y[k]/x[k]-1) for k in ['median_throughput','ttft_p50','ttft_p95','output_tps_p50','amortized_itl_p50','amortized_itl_p95']}
        result['bench'][str(bs)]=dict(baseline=x,native=y,change_percent=change,
            needs_regression_review=(change['median_throughput'] < -5 or change['ttft_p50']>5 or change['amortized_itl_p50']>5))
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

```
