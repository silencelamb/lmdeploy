# run_followup.py

原始验证文件快照；源码 SHA256：`11b9753939f13844f51c5005754a3d1838b01618220c4783245cc60b31293cc7`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
results=[]
def run(env,label,args,seed=None,script='pipeline_case.py',timeout=1800):
    cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',label,'--timeout',str(timeout)]
    if seed is not None: cmd+=['--global-seed',str(seed)]
    cmd+=['--','python',str(ROOT/'scripts'/script),*args,'--output',str(ROOT/'results'/f'{label}.json')]
    print('START',label,flush=True)
    rc=subprocess.run(cmd).returncode
    results.append(dict(label=label,returncode=rc))
    (ROOT/'results/followup-progress.json').write_text(json.dumps(results,indent=2))
    print('DONE',label,rc,flush=True)

for env in ['baseline','native']:
    run(env,env+'-sage-mtp3-eager-score-trace',['--eager','--mtp','3','--prefix','--scores','--suite','trace','--new-tokens','64'])
for env in ['baseline','native']:
    for rep in [0,1]:
        run(env,f'{env}-sage-mtp3-seeded-r{rep}',['--mtp','3','--prefix','--scores','--suite','trace','--new-tokens','64'],seed=42)
for env in ['baseline','native']:
    run(env,env+'-sage-mtp3-extra',['--mtp','3','--prefix','--scores','--suite','extra','--length','8192'])
for env in ['baseline','native']:
    run(env,env+'-sage-http-bench',['--repeats','5'],script='http_case.py')

```
