# freeze_ray_stall.py

```python
"""Intrusive one-shot native stacks after a real >100ms outstanding RPC.

Only used by a diagnostic run. All induced pause time is explicitly recorded.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('pids',nargs='+',type=int);a=p.parse_args()
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    try:pending=json.loads((a.root/'pending.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(.005);continue
    if pending.get('active') and time.monotonic_ns()-pending['dispatch_ns']>100_000_000 and pending['item']>10:
        record=dict(pending=pending,detected_ns=time.monotonic_ns(),pids=a.pids)
        for pid in a.pids:os.kill(pid,signal.SIGSTOP)
        record['all_stopped_ns']=time.monotonic_ns()
        (a.root/'freeze.json').write_text(json.dumps(record,indent=2))
        try:
            base=Path(__file__).resolve().parents[1]/'tools/local'
            env=os.environ.copy();env['LD_LIBRARY_PATH']=str(base/'usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
            for pid in a.pids:
                cmd=[str(base/'usr/bin/gdb'),'-nx','-batch','-p',str(pid),
                    '-ex','set debuginfod enabled off','-ex','set pagination off',
                    '-ex','thread apply all bt 14','-ex','detach']
                with (a.root/f'gdb-{pid}.log').open('w') as out:
                    subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=env,timeout=45)
        finally:
            for pid in a.pids:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
            record['resumed_ns']=time.monotonic_ns()
            (a.root/'freeze.json').write_text(json.dumps(record,indent=2))
        break
    time.sleep(.005)
```
