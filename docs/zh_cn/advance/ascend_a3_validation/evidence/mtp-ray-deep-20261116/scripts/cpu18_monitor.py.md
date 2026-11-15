# cpu18_monitor.py

```python
"""Read-only host CPU / softirq telemetry from a different CPU."""
import argparse
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=40)
a=p.parse_args();assert not a.output.exists(),a.output
os.sched_setaffinity(0,{42})
start=time.monotonic()
with a.output.open('x') as out:
    while time.monotonic()-start<a.seconds:
        row=dict(ns=time.monotonic_ns())
        for l in Path('/proc/stat').read_text().splitlines():
            x=l.split()
            if x and x[0] in ['cpu18','cpu42']:
                row[x[0]]=list(map(int,x[1:]))
        row['softirqs']={}
        for l in Path('/proc/softirqs').read_text().splitlines()[1:]:
            x=l.split();row['softirqs'][x[0]]=[int(x[19]),int(x[43])]
        out.write(json.dumps(row)+'\n');time.sleep(.02)
```
