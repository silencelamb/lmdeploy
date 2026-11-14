# setup_mooncake_envs.py

原始验证文件快照；源码 SHA256：`69b8c1ba8e74e26f3a54cd65b83dfb8319eda5b6100c5e24ef68daba9a5400b6`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Add separate Mooncake environments without changing the initial comparisons."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import venv

ROOT=Path(__file__).resolve().parents[1]
specs={'baseline-mooncake':('fb5de87a45df9999a19a569ed9441df8edfa6975','codex/ascend-a3-mooncake-baseline-20261112'),
       'native-mooncake':('6421cefea94edd350e419f556158639556357a27','codex/ascend-a3-mooncake-20261112')}
for name,(revision,branch) in specs.items():
    source=ROOT/'sources'/('lmdeploy-'+name)
    if not source.exists():
        subprocess.run(['git','-C',str(ROOT.parent/'lmdeploy'),'worktree','add','-b',branch,str(source),revision],check=True)
    dest=ROOT/'envs'/name
    assert not dest.exists(),dest
    venv.EnvBuilder(with_pip=True,system_site_packages=False).create(dest)
    site=dest/'lib/python3.12/site-packages'
    for p in (ROOT/'envs/dependencies').iterdir():
        if 'mooncake' in p.name.lower():
            continue
        target=site/p.name
        if not target.exists():
            target.symlink_to(p,target_is_directory=p.is_dir())
    original=ROOT/'envs'/name.split('-')[0]/'lib/python3.12/site-packages/sitecustomize.py'
    shutil.copy2(original,site/'sitecustomize.py')
    print('CREATED',name,source,flush=True)
wheel=next((ROOT/'artifacts/wheels').glob('mooncake_transfer_engine_npu-0.3.13.post1-cp312*.whl'))
(ROOT/'inventory/mooncake-wheel.json').write_text(json.dumps(dict(file=str(wheel),bytes=wheel.stat().st_size,
    sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),version='0.3.13.post1',
    source='https://pypi.org/project/mooncake-transfer-engine-npu/0.3.13.post1/'),indent=2))
for name in ['Agents-A1 A3平台-LMDeploy框架-推理部署方案.md','GLM-5.3-Flash LMDeploy框架线上部署方案-20261106-a0102d47967d.md']:
    shutil.copy2(ROOT.parent/name,ROOT/'inventory'/name)

```
