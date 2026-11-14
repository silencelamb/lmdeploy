# setup_envs.py

原始验证文件快照；源码 SHA256：`4d9a696d7b9c601e7ad271da5e589773f59291e6b03e2a83cec34c70e18c20dd`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Snapshot this container's dependencies without modifying the original installation."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import venv

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
COMMON = ROOT / 'envs/dependencies'
COMMON.mkdir(exist_ok=True)
excluded = []
for p in SITE.iterdir():
    if any(s in p.name.lower() for s in ('dlinfer', 'lmdeploy', 'triton_ascend_kernels')):
        excluded.append(p.name)
        continue
    target = COMMON / p.name
    if not target.exists():
        subprocess.run(['cp', '-a', '--reflink=auto', str(p), str(target)], check=True)
triton = ROOT / 'sources/triton-ascend-kernels-snapshot'
if not triton.exists():
    shutil.copytree('/root/workspace/code/triton-ascend-kernels', triton,
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))
# Preserve the exact modified external kernel source; do not build/edit colleague checkout.
(COMMON / 'triton_ascend_kernels_snapshot.pth').write_text(str(triton / 'src') + '\n')
for p in SITE.glob('triton_ascend_kernels-*.dist-info'):
    shutil.copytree(p, COMMON / p.name, dirs_exist_ok=True)
for name in ('baseline', 'native', 'standalone'):
    env = ROOT / 'envs' / name
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env)
    dest = env / 'lib/python3.12/site-packages'
    for p in COMMON.iterdir():
        target = dest / p.name
        if not target.exists():
            target.symlink_to(p, target_is_directory=p.is_dir())
    print('created', name, flush=True)
manifest = {}
for p in triton.rglob('*'):
    if p.is_file():
        manifest[str(p.relative_to(triton))] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / 'inventory/external-kernel-files.sha256.json').write_text(json.dumps(manifest, indent=2))
(ROOT / 'inventory/dependency-snapshot.json').write_text(json.dumps({
    'python': sys.version, 'source': str(SITE), 'common_snapshot': str(COMMON),
    'excluded': excluded, 'system_site_packages': False,
    'note': 'Independent venv package namespaces; common dependencies copied from container and linked read-only by convention. All project installs use --no-deps.',
}, indent=2))

```
