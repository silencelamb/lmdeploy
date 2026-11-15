# sched_all_monitor.py

```python
"""Read-only task scheduling samples; no ptrace, signals, priority changes."""
import json
from pathlib import Path
import sys
import time

root=Path(sys.argv[1]); tids=json.loads((root/'monitor-tids.json').read_text())
with (root/'sched.jsonl').open('x') as out:
    while not (root/'monitor-stop').exists():
        row=dict(ns=time.monotonic_ns(),tasks={})
        for tid in tids:
            try:
                raw=Path(f'/proc/{tid}/stat').read_text().rsplit(')',1)[1].split()
                stat=[int(v) for v in Path(f'/proc/{tid}/schedstat').read_text().split()]
                row['tasks'][tid]=dict(state=raw[0],runtime_ns=stat[0],runqueue_ns=stat[1],slices=stat[2],
                    utime=int(raw[11]),stime=int(raw[12]),nice=int(raw[16]),processor=int(raw[36]),
                    minflt=int(raw[7]),majflt=int(raw[9]))
                if raw[0] in ('R','D'):
                    try:
                        row['tasks'][tid]['wchan']=Path(f'/proc/{tid}/wchan').read_text().strip()
                        row['tasks'][tid]['syscall']=Path(f'/proc/{tid}/syscall').read_text().strip()
                    except (PermissionError,ProcessLookupError):pass
            except (FileNotFoundError,ProcessLookupError):pass
        out.write(json.dumps(row)+'\n')
        time.sleep(.02)
```
