# run_model_queue.py

原始验证文件快照；源码 SHA256：`c3c10b8368c45ecaacc57df85cf2427b7051d5ae646c201458e0bb5bf03d928c`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Serial paired workloads; stop on a failed run to investigate before proceeding."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
jobs=[('native','Agents-A1',0,'agents'),('baseline','Sage-A1.5',3,'sage'),('native','Sage-A1.5',3,'sage')]
records=[]
for env,model,mtp,short in jobs:
    label=f'{env}-{short}-external-pr5030'
    command=[sys.executable,'scripts/run.py','--env',env,'--label',label,'--devices','0,1,2,3','--timeout','1500',
        '--','env','ASCEND_ENABLE_USE_FABRIC_MEM=1','python','scripts/model_external_cache.py',
        '--model',f'/afs-weights/{model}','--mtp',str(mtp),'--label',label,'--output',f'results/{label}.json']
    print('START',label,flush=True)
    rc=subprocess.call(command,cwd=ROOT)
    records.append(dict(label=label,returncode=rc))
    (ROOT/'results/model-queue.json').write_text(json.dumps(records,indent=2))
    print('FINISH',label,rc,flush=True)
    if rc:raise SystemExit(rc)
    subprocess.run([sys.executable,'scripts/collect_runtime_logs.py'],cwd=ROOT,check=True)

```
