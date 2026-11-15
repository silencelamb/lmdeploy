# run_check.py

```python
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--no-vendor',action='store_true')
p.add_argument('--repo',default=str(root/'lmdeploy'))
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args()
venv = Path('/root/workspace/a3-review-20261115/envs/cpu-no-torch-npu') if a.no_vendor else Path('/root/workspace/a3-mooncake-pr5030-20261113/envs/native')
env=os.environ.copy()
env.update(PYTHONPATH=f'{root}/guard:{a.repo}:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe',
           PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
           PATH=str(venv/'bin')+':'+env['PATH'],VIRTUAL_ENV=str(venv),
           VALIDATION_ROOT=str(root),VALIDATION_ENV='review',
           ASCEND_REVIEW_NO_VENDOR='1' if a.no_vendor else '0',
           TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',
           HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
args=a.args[1:] if a.args and a.args[0]=='--' else a.args
cmd=[str(venv/'bin/python'),*args]
record=dict(label=a.label,command=cmd,cwd=a.repo,no_vendor=a.no_vendor,
            start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),
            environment={k:env[k] for k in ('PYTHONPATH','PYTHONNOUSERSITE','TORCH_DEVICE_BACKEND_AUTOLOAD','ASCEND_REVIEW_NO_VENDOR')})
log=root/'logs'/f'{a.label}.log'
assert not log.exists(),log
start=time.monotonic()
with log.open('w') as f:
 result=subprocess.run(cmd,cwd=a.repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
record.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start)
(root/'logs'/f'{a.label}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
print('\n'.join(log.read_text().splitlines()[-18:]))
raise SystemExit(result.returncode)
```
