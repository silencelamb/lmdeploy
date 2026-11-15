# run_lint.py

```python
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
repo=root/'lmdeploy'
files=json.loads((root/'inventory/changed-files.json').read_text())
env=os.environ.copy()
env['PYTHONPATH']=''
env['PRE_COMMIT_HOME']=str(root/'artifacts/pre-commit-cache')
urls=['https://github.com/astral-sh/ruff-pre-commit','https://github.com/pre-commit/pre-commit-hooks',
      'https://github.com/executablebooks/mdformat','https://github.com/myint/docformatter',
      'https://github.com/open-mmlab/pre-commit-hooks','https://github.com/pre-commit/mirrors-clang-format']
env['GIT_CONFIG_COUNT']=str(len(urls))
for i,url in enumerate(urls):
 env[f'GIT_CONFIG_KEY_{i}']=f'url.{url}.insteadOf'
 env[f'GIT_CONFIG_VALUE_{i}']=url
cmd=['/root/workspace/a3-pr-prep-20261115/lint-env/bin/pre-commit','run','--config',str(root/'artifacts/applicable-pre-commit.yaml'),'--files',*files]
label=sys.argv[1]
log=root/'logs'/f'{label}.log'
start=time.monotonic()
with log.open('w') as f:
 result=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
(root/'logs'/f'{label}.json').write_text(json.dumps({'command':cmd,'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
print(log.read_text()[-16000:])
raise SystemExit(result.returncode)
```
