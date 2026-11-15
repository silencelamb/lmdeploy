# run_s2.py

```python
"""Retry both Intern-S2 families with verified RDKit import ordering."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
records=[]
for name, model in [('s2preview','Intern-S2-Preview'),('s2mobius','Intern-S2-Mobius')]:
    for side in ('before','after'):
        label=f'{name}-rdkit-all-{side}'
        repo=str(ROOT/'lmdeploy-before') if side=='before' else '/root/workspace/a3-native-pr-20261115/lmdeploy'
        cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--label',label,'--repo',repo,'--devices','0,1',
             '--timeout','900','--',str(ROOT/'scripts/pipeline_case.py'),'--model',f'/afs-weights/{model}',
             '--tp','2','--scores','--prefill','256','--session','8192','--batch','8','--cache-fraction','0.5',
             '--preload-rdkit','--output',str(ROOT/'results'/f'{label}.json')]
        print('START',label,flush=True)
        with (ROOT/'logs'/f'{label}-runner.log').open('w') as f:
            rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        records.append(dict(label=label,returncode=rc,command=cmd))
        (ROOT/'results/s2-rdkit-all-runs.json').write_text(json.dumps(records,indent=2)+'\n')
        print('FINISH',label,rc,flush=True)
```
