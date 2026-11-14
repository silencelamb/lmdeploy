# install_tilelang.py

原始验证文件快照；源码 SHA256：`aec3b2c1fa8fdb4fed346be2a38837ab161f49c8e9a4379c3d168e79a9b65be0`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Install a verified, pinned dependency bundle into both isolated namespaces."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
selected=json.loads((ROOT/'inventory/tilelang-followup/selected-wheels.json').read_text())
wheels=[]
for item in selected:
    path=ROOT/'artifacts/tilelang-wheels'/item['filename']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    wheels.append(str(path))
bundle=ROOT/'envs/tilelang-0.1.15-dependencies'
assert not bundle.exists(), 'Do not overwrite an existing dependency bundle'
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(bundle),*wheels],check=True)
record={}
for name in ['baseline','native']:
    site=ROOT/'envs'/name/'lib/python3.12/site-packages'
    removed={};added={}
    old_metadata=list(site.glob('apache_tvm_ffi-*.dist-info'))
    for target in old_metadata:
        assert target.is_symlink(),target
        removed[target.name]=str(target.resolve());target.unlink()
    for source in bundle.iterdir():
        if source.name in ['bin','__pycache__']:continue
        target=site/source.name
        if target.exists() or target.is_symlink():
            assert target.is_symlink(),target
            removed[target.name]=str(target.resolve());target.unlink()
        target.symlink_to(source,target_is_directory=source.is_dir())
        added[target.name]=str(source)
    record[name]=dict(removed_symlinks=removed,added_symlinks=added)
(ROOT/'inventory/tilelang-followup/environment-changes.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))

```
