# summarize.py

原始验证文件快照；源码 SHA256：`df1bb3a956f3132016041b51ed5a378b6cab814c33e96404ffdf4d3873debc8f`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Machine-readable index of this follow-up, retaining failures as failures."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
result=dict(revisions=read('inventory/comparison-revisions.json'),
    pr=read('inventory/pr5030-info.json'),sdk={},comparisons={},commands={},unit_suites={},
    local_buffer_controls=read('results/local-buffer-control-summary.json'),
    preservation=read('inventory/preservation-final.json'),cleanup=read('inventory/cleanup-final.json'))
for path in (ROOT/'results').glob('*fabric-sdk*.json'):
    result['sdk'][path.stem]=json.loads(path.read_text())
for path in (ROOT/'results').glob('compare-*-external-pr5030*.json'):
    result['comparisons'][path.stem]=json.loads(path.read_text())
for path in (ROOT/'logs').glob('*.command.json'):
    data=json.loads(path.read_text())
    result['commands'][path.stem]={k:data.get(k) for k in ['label','env','returncode','elapsed_seconds','source_revisions']}
for name in ['baseline','native']:
    path=ROOT/'results'/f'{name}-pr5030-worker-unit-tilelang.xml'
    cases=list(ET.parse(path).iter('testcase'))
    failures=[dict(name=case.get('name'),message=case.find('failure').get('message')) for case in cases if case.find('failure') is not None]
    result['unit_suites'][name]=dict(source=str(path.relative_to(ROOT)),tests=len(cases),failures=failures,
        errors=[case.get('name') for case in cases if case.find('error') is not None],
        skipped=[case.get('name') for case in cases if case.find('skipped') is not None],
        passed=sum(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in cases))
result['tilelang_followup']=dict(
    selected_wheels=read('inventory/tilelang-followup/selected-wheels.json'),
    audits={name:read(f'inventory/tilelang-followup/{name}-audit.json') for name in ['baseline','native']},
    initial_results={name:f'results/{name}-pr5030-worker-unit.xml' for name in ['baseline','native']},
    original_archive='evidence-before-tilelang.tar.gz')
rows=[row for comp in result['comparisons'].values() for role in comp['migration_comparison'].values() for row in role.values()]
result['paired_numeric_totals']=dict(sequences=len(rows),exact_sequences=sum(row['exact_tokens'] for row in rows),
    compared_scores=sum(row['score_count'] for row in rows),max_abs=max(row['max_abs'] for row in rows))
result['scope_limits']=['This follow-up supersedes the previous Mooncake host-blocker interpretation.',
    'Previous Sage MTP long-generation strict equivalence failures and HTTP C4 degradation signals remain unresolved.',
    'C1 short retrieval is not a long-generation, C4/C16, multi-node, DP/EP, GLM, quantization or production-reliability certificate.',
    'Original image dependency metadata/pip check remains unsatisfied.',
    'The previous missing-tilelang CPU test failure is resolved in both environments; this does not cover GLM full-model inference or CUDA kernel execution.']
(ROOT/'results/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(result['paired_numeric_totals'])

```
