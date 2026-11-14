# run_http_recheck.py

原始验证文件快照；源码 SHA256：`fa757f5b5c5aba93e7c9be55f117213c81cc6d2f8fde84cccacfcfb83cd0ace3`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Reverse the first HTTP pair's order and preserve device telemetry."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
progress = []
for env in ['native', 'baseline']:
    name = env + '-sage-http-recheck'
    cmd = [sys.executable, str(ROOT/'scripts/run.py'), '--env', env,
           '--label', name, '--timeout', '1200', '--', 'python',
           str(ROOT/'scripts/http_case.py'), '--repeats', '7',
           '--output', str(ROOT/'results'/f'{name}.json')]
    print('START', name, flush=True)
    child = subprocess.Popen(cmd)
    with (ROOT/'logs'/f'{name}-npu.jsonl').open('w') as telemetry:
        while child.poll() is None:
            sample = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True)
            telemetry.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                returncode=sample.returncode, stdout=sample.stdout, stderr=sample.stderr))+'\n')
            telemetry.flush()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    progress.append(dict(name=name, returncode=child.returncode))
    (ROOT/'results/http-recheck-progress.json').write_text(json.dumps(progress, indent=2))
    print('DONE', name, child.returncode, flush=True)

```
