# run_agents_http.py

原始验证文件快照；源码 SHA256：`ab9780cdef5ea7e6db20f6c09f1430e1f38d6ecddbc725c2bda9b7008181784d`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
progress=[]
for env in ['baseline','native']:
    name=env+'-agents-http'
    cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',name,'--timeout','1800','--',
         'python',str(ROOT/'scripts/http_case.py'),'--model','/afs-weights/Agents-A1',
         '--model-name','agents-a1','--mtp','0','--repeats','5','--output',str(ROOT/'results'/f'{name}.json')]
    print('START',name,flush=True);rc=subprocess.run(cmd).returncode
    progress.append(dict(name=name,returncode=rc))
    (ROOT/'results/agents-http-progress.json').write_text(json.dumps(progress,indent=2))
    print('DONE',name,rc,flush=True)

```
