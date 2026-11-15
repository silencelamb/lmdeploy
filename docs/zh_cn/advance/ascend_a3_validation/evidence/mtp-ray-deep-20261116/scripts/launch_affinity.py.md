# launch_affinity.py

```python
"""Exclude explicitly selected host CPUs for one command and its children.

No system IRQ settings or Ray scheduling/synchronization are changed.
Example: python launch_affinity.py --exclude 18 -- python scripts/run_deep.py ...
"""
import argparse
import json
import os

p=argparse.ArgumentParser();p.add_argument('--exclude',type=int,nargs='+',required=True)
p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
assert cmd,'A command is required'
before=os.sched_getaffinity(0);after=before-set(a.exclude)
assert after,'Exclusion would leave no allowed CPU'
os.sched_setaffinity(0,after)
print(json.dumps(dict(original_affinity=sorted(before),excluded=a.exclude,effective_affinity=sorted(after),command=cmd)),flush=True)
os.execvpe(cmd[0],cmd,os.environ)
```
