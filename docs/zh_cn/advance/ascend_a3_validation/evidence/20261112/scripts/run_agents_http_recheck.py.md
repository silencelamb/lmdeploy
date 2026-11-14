# run_agents_http_recheck.py

原始验证文件快照；源码 SHA256：`2b334cf171b18e0482dfc2a88235dd3c78390db5774a2e55b4827458e2fa7b0b`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Reverse order for the Agents HTTP review, after isolated Store attempts."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+1800
while True:
    p=ROOT/'results/mooncake-model-progress.json'
    if p.exists() and len(json.loads(p.read_text()))==4:break
    if time.monotonic()>deadline:raise TimeoutError('Store model attempts did not finish')
    time.sleep(2)
progress=[]
for env in ['native','baseline']:
    name=env+'-agents-http-recheck'
    cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',name,'--timeout','1200','--',
         'python',str(ROOT/'scripts/http_case.py'),'--model','/afs-weights/Agents-A1',
         '--model-name','agents-a1','--mtp','0','--repeats','7','--concurrency','1','4',
         '--output',str(ROOT/'results'/f'{name}.json')]
    print('START',name,flush=True);rc=subprocess.run(cmd).returncode
    progress.append(dict(name=name,returncode=rc))
    (ROOT/'results/agents-http-recheck-progress.json').write_text(json.dumps(progress,indent=2))
    print('DONE',name,rc,flush=True)

```
