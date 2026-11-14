# run_buffer_controls.py

原始验证文件快照；源码 SHA256：`60016c6db2878b03cf008941475d0ee68dc6361ce086d4f383524ecce0c6c900`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Wait for timing tests to finish, then isolate the local-buffer setup error."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+2400
while True:
    p=ROOT/'results/agents-recheck-queue.json'
    rows=json.loads(p.read_text()) if p.exists() else []
    if any(x['returncode'] for x in rows):raise SystemExit('recheck failed')
    if len(rows)==2:break
    if time.monotonic()>deadline:raise SystemExit('recheck did not finish')
    time.sleep(2)
out=[]
for size in [0,256]:
    label=f'native-local-buffer-{size}m-control'
    argv=[sys.executable,'scripts/run.py','--env','native','--label',label,'--devices','6','--timeout','120',
        '--','env','ASCEND_ENABLE_USE_FABRIC_MEM=1','python','scripts/local_buffer_control.py',
        '--local-mib',str(size),'--output',f'results/{label}.json']
    rc=subprocess.call(argv,cwd=ROOT)
    result=json.loads((ROOT/'results'/f'{label}.json').read_text())
    log=(ROOT/'logs'/f'{label}.log').read_text(errors='replace')
    out.append(dict(local_mib=size,returncode=rc,setup_success=result['setup_success'],
        map_mem_failure='Map fabric memory failed' in log,source=f'results/{label}.json'))
(ROOT/'results/local-buffer-control-summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

```
