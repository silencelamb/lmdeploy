"""Create an isolated Python/Triton copy; never invoke pip over linked deps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/root/workspace/a3-mooncake-pr5030-20261113/envs/native')
p = argparse.ArgumentParser()
p.add_argument('--name', default='upgrade5-ir2')
p.add_argument('--wheel-root', type=Path)
a = p.parse_args()
dest = ROOT / 'envs' / a.name
assert not dest.exists(), dest
subprocess.run(['/usr/local/python3.12.13/bin/python3.12', '-m', 'venv', '--copies', '--without-pip', str(dest)], check=True)
src = BASE / 'lib/python3.12/site-packages'
out = dest / 'lib/python3.12/site-packages'
records = []
for item in sorted(src.iterdir()):
    if 'dlinfer' in item.name.lower():
        raise RuntimeError('Unexpected dlinfer dependency: ' + str(item))
    if (item.name.startswith('__editable__') or item.name == '__pycache__'
            or (a.wheel_root and (a.wheel_root / item.name).exists())
            or (a.wheel_root and item.name.startswith('triton_ascend-'))):
        records.append(dict(name=item.name, mode='excluded'))
    elif item.name == 'triton':
        shutil.copytree(item.resolve(), out / item.name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        records.append(dict(name=item.name, mode='independent-copy', source=str(item.resolve())))
    else:
        (out / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
        records.append(dict(name=item.name, mode='read-only-reuse-no-pip', source=str(item.resolve())))
if a.wheel_root:
    for item in sorted(a.wheel_root.iterdir()):
        assert not (out / item.name).exists()
        (out / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
        records.append(dict(name=item.name, mode='extracted-official-wheel', source=str(item.resolve())))
manifest = dict(baseline=str(BASE), destination=str(dest), entries=records)
(ROOT / 'inventory' / f'{a.name}-creation.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(dest)
