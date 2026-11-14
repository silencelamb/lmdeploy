# run_agents_cases.py

原始验证文件快照；源码 SHA256：`baed5d7607c08d06177de2e645004a077468be1bda494b3c9dd566fb2fd08a14`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
progress=[]
cases=[('agents-eager-trace',['--eager','--scores','--suite','trace']),
       ('agents-graph-prefix-extra',['--prefix','--scores','--suite','extra','--length','8192']),
       ('agents-context-boundary',['--prefix','--scores','--suite','long','--length','262000','--no-thinking','--respect-eos'])]
for label,args in cases:
    for env in ['baseline','native']:
        name=env+'-'+label
        cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',name,'--timeout','1200',
             '--','python',str(ROOT/'scripts/pipeline_case.py'),'--model','/afs-weights/Agents-A1',
             *args,'--output',str(ROOT/'results'/f'{name}.json')]
        print('START',name,flush=True)
        rc=subprocess.run(cmd).returncode
        progress.append(dict(name=name,returncode=rc))
        (ROOT/'results/agents-progress.json').write_text(json.dumps(progress,indent=2))
        print('DONE',name,rc,flush=True)

```
