# run_agents_pipeline_control.py

原始验证文件快照；源码 SHA256：`d014f0eb1fb0329b2279c621be3a7b5e65a5de5680c4db2fd260cf5445c5a8ed`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
progress=[]
for env in ['baseline','native']:
    label=env+'-agents-c4-pipeline-control'
    cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',label,'--timeout','900','--',
        'python',str(ROOT/'scripts/pipeline_case.py'),'--model','/afs-weights/Agents-A1','--prefix',
        '--new-tokens','128','--suite','bench','--bench-concurrency','4','--bench-http-workload',
        '--no-thinking','--repeats','7','--output',str(ROOT/'results'/f'{label}.json')]
    print('START',label,flush=True);rc=subprocess.run(cmd).returncode
    progress.append(dict(label=label,returncode=rc))
    (ROOT/'results/agents-pipeline-control-progress.json').write_text(json.dumps(progress,indent=2))
    print('DONE',label,rc,flush=True)

```
