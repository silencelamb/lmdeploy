# package_evidence.py

原始验证文件快照；源码 SHA256：`77c78482c53b92e09c4170feb3f2c258b5d8bcb26efc135c0870ae0e26378462`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Package reviewable evidence, retaining source/env directories on this machine."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
paths=set()
for name in ['ACCEPTANCE.txt','ACCEPTANCE-EXTENSION.txt','REPRODUCE.txt','REPORT.txt']:
    paths.add(ROOT/name)
for name in ['scripts','logs','results','inventory','configs','artifacts/harness-snapshots','artifacts/wheels']:
    for p in (ROOT/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:paths.add(p)
for p in (ROOT/'artifacts').iterdir():
    if p.is_file() and p.suffix in ['.patch','.txt']:paths.add(p)
for rel in json.loads((ROOT/'inventory/external-kernel-files.sha256.json').read_text()):
    paths.add(ROOT/'sources/triton-ascend-kernels-snapshot'/rel)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
index=ROOT/'EVIDENCE-SHA256.json';index.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for p in sorted(paths|{index}):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(ROOT/'evidence.tar.gz.sha256').write_text(digest+'  evidence.tar.gz\n')
print(json.dumps(dict(files=len(manifest),archive=str(archive),bytes=archive.stat().st_size,sha256=digest),indent=2))

```
