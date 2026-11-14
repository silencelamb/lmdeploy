# analyze_external_cache.py

原始验证文件快照；源码 SHA256：`7e7af04f7738d609ae3cabe68240e19ace61606a569059a33107a1344220208c`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Compare exact histories and distinguish external Store reads from local hits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]

def compare_rows(a,b):
    assert a['prompt']==b['prompt'], 'different prompts are not comparable'
    errors=[];compared=0
    for i,(left,right) in enumerate(zip(a['logprobs'],b['logprobs'])):
        errors.extend(abs(left[k]-right[k]) for k in left.keys() & right.keys())
        compared+=1
        if i>=len(a['token_ids']) or i>=len(b['token_ids']) or a['token_ids'][i]!=b['token_ids'][i]:break
    max_abs=max(errors) if errors else None
    rms=math.sqrt(sum(v*v for v in errors)/len(errors)) if errors else None
    return dict(exact_tokens=a['token_ids']==b['token_ids'],same_input_length=a['input_token_len']==b['input_token_len'],
        score_count=len(errors),common_history_steps=compared,max_abs=max_abs,rms=rms,
        numerical_pass=bool(errors) and max_abs<=.02 and rms<=.005)

def inspect(path):
    data=json.loads(path.read_text());rows={role:{} for role in ['producer','consumer']}
    for item in data['events']:
        if item['event']=='response':rows[item['role']][item['response']['label']]=item['response']
    result=dict(complete=data['complete'],source=str(path),cold_to_external={},logs={},
        child_exitcodes=data.get('child_exitcodes'),lifecycle_pass=data.get('child_exitcodes')==[0,0])
    for label in rows['producer'].keys() & rows['consumer'].keys():
        a=rows['producer'][label];b=rows['consumer'][label]
        result['cold_to_external'][label]=dict(compare_rows(a,b),input_tokens=a['input_token_len'],
            cold_cached=a['cached_tokens'],external_cached=b['cached_tokens'],
            correct=a['expected'] in a['text'] and b['expected'] in b['text'],warmup=a['warmup'],
            cold_ttft=a['ttft'],external_ttft=b['ttft'],cold_elapsed=a['elapsed'],external_elapsed=b['elapsed'])
    for role in rows:
        log_path=ROOT/'logs'/f"{data['options']['label']}-{role}.log"
        raw_dir=ROOT/'artifacts/ray-logs'/data['options']['label']/role
        raw_files=list(raw_dir.glob('worker-*'))
        log=('\n'.join(p.read_text(errors='replace') for p in raw_files) if raw_files
             else log_path.read_text(errors='replace') if log_path.exists() else '')
        log=re.sub(r'\x1b\[[0-9;]*m','',log)
        saves=re.findall(r'operation=save_batch_put_from_multi_buffers[^\n]+status=ok[^\n]*',log)
        loads=re.findall(r'operation=load_batch_get_into_multi_buffers[^\n]+status=ok[^\n]*',log)
        failures=[line for line in log.splitlines() if 'Mooncake Store interaction after:' in line and ('status=error' in line or 'status=partial_failure' in line)]
        result['logs'][role]=dict(successful_save_batches=len(saves),successful_load_batches=len(loads),
            load_records=loads,failures=failures,raw_worker_files=[str(p.relative_to(ROOT)) for p in raw_files],
            driver_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None)
    pairs=list(result['cold_to_external'].values())
    result['external_function_pass']=bool(pairs) and data['complete'] and all(x['correct'] and x['external_cached']>0 for x in pairs) and result['logs']['consumer']['successful_load_batches']>=2*len(pairs) and not any(x['failures'] for x in result['logs'].values())
    result['strict_precision_pass']=bool(pairs) and all(x['exact_tokens'] and x['numerical_pass'] for x in pairs)
    measured=[x for x in pairs if not x['warmup']]
    if measured:
        result['timing']={key:statistics.median(x[key] for x in measured) for key in ['cold_ttft','external_ttft','cold_elapsed','external_elapsed']}
        result['timing']['samples']=len(measured)
        result['timing']['external_ttft_speedup']=result['timing']['cold_ttft']/result['timing']['external_ttft']
    return result,rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('baseline',type=Path);parser.add_argument('native',type=Path)
    parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
    baseline,brows=inspect(a.baseline);native,nrows=inspect(a.native)
    bd=json.loads(a.baseline.read_text());nd=json.loads(a.native.read_text())
    assert bd['store_config']==nd['store_config'], 'different Store parameters'
    for key in ['model','mtp','segment_gib','repeats']:
        assert bd['options'][key]==nd['options'][key],key
    for role in ['producer','consumer']:
        bc=next(e['config'] for e in bd['events'] if e['event']=='ready' and e['role']==role)
        nc=next(e['config'] for e in nd['events'] if e['event']=='ready' and e['role']==role)
        for config in [bc,nc]:
            extra=config['kv_transfer_config']['kv_connector_extra_config']
            extra.pop('mooncake_config_path');extra.pop('cache_prefix')
        assert bc==nc,'different engine configurations'
    cross={}
    for role in brows:
        cross[role]={label:compare_rows(brows[role][label],nrows[role][label]) for label in brows[role].keys() & nrows[role].keys()}
    result=dict(baseline=baseline,native=native,migration_comparison=cross)
    result['migration_precision_pass']=bool(cross['consumer']) and baseline['complete'] and native['complete'] and all(row['exact_tokens'] and row['numerical_pass'] for rows in cross.values() for row in rows.values())
    if 'timing' in baseline and 'timing' in native:
        result['timing_change_percent']={key:(native['timing'][key]/baseline['timing'][key]-1)*100 for key in ['cold_ttft','external_ttft','cold_elapsed','external_elapsed']}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

```
