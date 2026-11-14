# run_agents_recheck.py

原始验证文件快照；源码 SHA256：`5250119594c7efb0218cc0dec6736716214dcbd4328db6fa8c541baebb535868`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Reverse order, larger fresh-prefix sample after the first +6.0% TTFT signal."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+2400
while True:
    progress=ROOT/'results/model-queue.json'
    rows=json.loads(progress.read_text()) if progress.exists() else []
    if any(row['returncode'] for row in rows):raise SystemExit('initial model queue failed')
    if len(rows)==3:break
    if time.monotonic()>deadline:raise SystemExit('initial model queue did not finish')
    time.sleep(2)
records=[]
for env in ['native','baseline']:
    label=f'{env}-agents-external-pr5030-recheck'
    argv=[sys.executable,'scripts/run.py','--env',env,'--label',label,'--devices','0,1,2,3','--timeout','1500',
        '--','env','ASCEND_ENABLE_USE_FABRIC_MEM=1','python','scripts/model_external_cache.py',
        '--model','/afs-weights/Agents-A1','--mtp','0','--repeats','15','--label',label,'--output',f'results/{label}.json']
    print('START',label,flush=True);rc=subprocess.call(argv,cwd=ROOT)
    records.append(dict(label=label,returncode=rc))
    (ROOT/'results/agents-recheck-queue.json').write_text(json.dumps(records,indent=2))
    print('FINISH',label,rc,flush=True)
    if rc:raise SystemExit(rc)
    subprocess.run([sys.executable,'scripts/collect_runtime_logs.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'scripts/analyze_external_cache.py',
    'results/baseline-agents-external-pr5030-recheck.json','results/native-agents-external-pr5030-recheck.json',
    '--output','results/compare-agents-external-pr5030-recheck.json'],cwd=ROOT,check=True)

```
