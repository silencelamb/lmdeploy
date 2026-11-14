# summarize.py

原始验证文件快照；源码 SHA256：`299ffab68d65d3dc61fa6c77ccf7a9c15649d9ecd87016b8fd082e7825b06422`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Build a machine-readable index without turning skips or incomplete runs into passes."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
comparisons={}
for p in sorted((ROOT/'results').glob('compare-*.json')):
    d=json.loads(p.read_text())
    if 'token_pass' in d:
        comparisons[p.stem]={k:d.get(k) for k in ['baseline','native','complete','sequences','exact_sequences','tokens','numerics','token_pass','numerical_status','fixed_history']}
suites={}
for label in ['native-cpu-contracts-importfix','standalone-cpu-contracts','baseline-operator-tests-logical2','native-operator-tests','standalone-operator-tests']:
    p=ROOT/'results'/f'{label}.xml'
    cases=list(ET.parse(p).iter('testcase'))
    suites[label]=dict(total=len(cases),passed=sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases),
        failed=[dict(name=c.get('name'),message=(c.find('failure') if c.find('failure') is not None else c.find('error')).get('message')) for c in cases if c.find('failure') is not None or c.find('error') is not None],
        skipped=[c.get('name') for c in cases if c.find('skipped') is not None],
        cpu_ring_cases=sum('causal_conv1d_ring' in c.get('classname','') for c in cases))
commands={}
for p in sorted((ROOT/'logs').glob('*.command.json')):
    d=json.loads(p.read_text());commands[d['label']]={k:d.get(k) for k in ['env','returncode','elapsed_seconds','start_utc','end_utc']}
payload=dict(revisions=json.loads((ROOT/'inventory/comparison-revisions.json').read_text()),comparisons=comparisons,test_suites=suites,commands=commands,
    explicit_limits=['MTP strict token/logprob equivalence did not pass; baseline itself is nonrepeatable.',
        'Three MLA hardware failures reproduce on baseline, native and standalone.',
        'MTP input-logprob scoring crashes the shared sequence path on both branches.',
        'Package dependency metadata / pip check do not pass in the supplied image.',
        'Unrun: GPQA, multi-node/DP/EP, quantized models, vision, CUDA, four-instance router, production-scale KV-copy stress.',
        'Unrun: complete-model random/mixed sampling and bad words, optional Triton full-model rejection, profiler trace, supported-version clean wheel installation.',
        'Both models show repeated C4 HTTP streaming output-rate degradation; the Agents fixed-batch pipeline control does not reproduce it. Cause remains unresolved.',
        'Mooncake NPU 0.3.13.post1 installs, but all four model startup attempts and both fabric SDK probes fail before cache transfer. External hits, data correctness and performance remain unverified.'])
payload['http']={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('compare-http-*.json'))}
payload['cross_mode']=json.loads((ROOT/'results/cross-mode-audit.json').read_text())
payload['preservation']=json.loads((ROOT/'inventory/preservation-final.json').read_text())
payload['cleanup']=json.loads((ROOT/'inventory/cleanup-final.json').read_text())
payload['agents_c4_pipeline_control']=dict(
    comparison=json.loads((ROOT/'results/compare-agents-c4-pipeline-control.json').read_text()),
    output_audit=json.loads((ROOT/'results/agents-c4-control-output-audit.json').read_text()))
mc_revisions=ROOT/'inventory/mooncake-comparison-revisions.json'
if mc_revisions.exists():
    payload['mooncake_revisions']=json.loads(mc_revisions.read_text())
    payload['mooncake_sdk']={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*-mooncake-sdk-*.json'))}
    payload['mooncake_models']={}
    for p in sorted((ROOT/'results').glob('*-fabric-model.json')):
        d=json.loads(p.read_text());label=p.stem
        command=json.loads((ROOT/'logs'/f'{label}.command.json').read_text())
        log=(ROOT/'logs'/f'{label}.log').read_text(errors='replace')
        payload['mooncake_models'][label]=dict(complete=d['complete'],returncode=command.get('returncode'),
            source_revisions=command.get('source_revisions'),
            fabric_map_failure='Map fabric memory failed' in log,
            model=d['options']['model'],mtp=d['options']['mtp'],events=len(d['events']))
(ROOT/'results/summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
with (ROOT/'results/numerical-comparisons.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['comparison','complete','exact_sequences','sequences','tokens','max_score_difference','rms_score_difference','numerical_status'])
    for name,d in comparisons.items():
        writer.writerow([name,d['complete'],d['exact_sequences'],d['sequences'],d['tokens'],d['numerics']['max_abs'],d['numerics']['rms'],d['numerical_status']])
print('Saved summary.json and numerical-comparisons.csv')

```
