# setup_envs.py

原始验证文件快照；源码 SHA256：`35aa67786a6c967d468b76eae7cd902cc92c7eb12a95e850e01ed583418db5a3`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Reuse frozen dependencies in new namespaces, preserving the preceding run."""
import json
from pathlib import Path
import shutil
import venv

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/workspace/a3-validation-20261112')
records={}
for name in ['baseline','native']:
    previous=OLD/'envs'/f'{name}-mooncake'/'lib/python3.12/site-packages'
    env=ROOT/'envs'/name
    venv.EnvBuilder(with_pip=True,system_site_packages=False).create(env)
    dest=env/'lib/python3.12/site-packages'
    skipped=[]
    for item in previous.iterdir():
        if any(s in item.name.lower() for s in ['lmdeploy','dlinfer','sitecustomize','__pycache__']):
            skipped.append(item.name)
            continue
        target=dest/item.name
        if not target.exists():target.symlink_to(item.resolve(),target_is_directory=item.is_dir())
    seed=(ROOT/'scripts/seed_startup.py').read_text()
    guard=(ROOT/'scripts/sitecustomize_native.py').read_text() if name=='native' else ''
    (dest/'sitecustomize.py').write_text(guard+'\n'+seed)
    records[name]=dict(dependencies_from=str(previous),excluded=skipped,system_site_packages=False)
(ROOT/'inventory/environment-namespaces.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))

```
