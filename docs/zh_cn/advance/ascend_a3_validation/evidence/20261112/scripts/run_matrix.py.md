# run_matrix.py

原始验证文件快照；源码 SHA256：`65132d6a63ed25512b1d286a18cb49be8f087c2842c191bd00fec251c3eb8035`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Sequential paired scenarios; never overlap accelerator performance workloads."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CASES=[
 ('sage-graph-prefix-extra',['--prefix','--scores','--suite','extra','--length','8192']),
 ('sage-eager-score-trace',['--eager','--scores','--suite','trace','--new-tokens','64']),
 ('sage-graph-chunk256-score-trace',['--scores','--prefill','256','--suite','trace','--new-tokens','64']),
 ('dense-eager-score-trace',['--model','/afs-weights/Qwen3-8B','--tp','1','--eager','--scores','--prefill','256','--session','32768','--cache-fraction','.8','--suite','trace','--new-tokens','64']),
 ('dense-graph-prefix-extra',['--model','/afs-weights/Qwen3-8B','--tp','1','--prefix','--scores','--session','32768','--cache-fraction','.8','--suite','extra','--length','8192']),
 ('sage-mtp1-score-trace',['--mtp','1','--prefix','--scores','--suite','trace','--new-tokens','64']),
 ('sage-mtp3-score-trace',['--mtp','3','--prefix','--scores','--suite','trace','--new-tokens','64']),
 ('sage-mtp3-long260k',['--mtp','3','--prefix','--no-thinking','--suite','long','--length','260000','--new-tokens','64']),
]
results=[]
for label,args in CASES:
    for env in ['baseline','native']:
        name=env+'-'+label
        cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',name,'--timeout','2400','--','python',str(ROOT/'scripts/pipeline_case.py'),*args,'--output',str(ROOT/'results'/f'{name}.json')]
        print('START',name,flush=True)
        rc=subprocess.run(cmd).returncode
        results.append(dict(name=name,returncode=rc))
        (ROOT/'results/matrix-progress.json').write_text(json.dumps(results,indent=2))
        print('DONE',name,rc,flush=True)
    # Comparison emits a structured failure even if generation tokens differ.
    old=ROOT/'results'/f'baseline-{label}.json';new=ROOT/'results'/f'native-{label}.json'
    if all(json.loads(p.read_text()).get('complete') for p in [old,new] if p.exists()):
        with (ROOT/'logs'/f'compare-{label}.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/compare.py'),str(old),str(new),'--output',str(ROOT/'results'/f'compare-{label}.json')],stdout=log,stderr=subprocess.STDOUT)

```
