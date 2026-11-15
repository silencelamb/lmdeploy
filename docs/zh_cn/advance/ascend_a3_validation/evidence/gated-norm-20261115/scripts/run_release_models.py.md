# run_release_models.py

```python
"""Final-tree output traces plus a longer Agents performance confirmation."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
jobs=[('sage-release-after','Sage-A1.5','after','trace',64,3),
      ('agents-release-after','Agents-A1','after','trace',64,3),
      ('agents-long-after','Agents-A1','after','bench',256,5),
      ('agents-long-before','Agents-A1','before','bench',256,5)]
records=[]
for label,model,side,suite,tokens,repeats in jobs:
    repo=str(ROOT/'lmdeploy-before') if side=='before' else '/root/workspace/a3-native-pr-20261115/lmdeploy'
    cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--label',label,'--repo',repo,'--devices','0,1',
         '--timeout','900','--',str(ROOT/'scripts/pipeline_case.py'),'--model',f'/afs-weights/{model}',
         '--tp','2','--scores','--prefill','256','--session','8192','--batch','8','--cache-fraction','0.5',
         '--suite',suite,'--bench-concurrency','1','4','--repeats',str(repeats),'--new-tokens',str(tokens),
         '--output',str(ROOT/'results'/f'{label}.json')]
    print('START',label,flush=True)
    with (ROOT/'logs'/f'{label}-runner.log').open('w') as f:
        rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    records.append(dict(label=label,returncode=rc,command=cmd))
    (ROOT/'results/release-model-runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print('FINISH',label,rc,flush=True)
```
