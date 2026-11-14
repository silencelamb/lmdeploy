# package_evidence.py

原始验证文件快照；源码 SHA256：`a7cef02fbe1470de52e2b3c6b7160929281e60ec7724b3afe8f74316f40402bc`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Archive follow-up evidence while retaining source worktrees and venvs here."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
paths={ROOT/name for name in ['ACCEPTANCE.txt','REPRODUCE.txt','REPORT.txt']}
for name in ['scripts','inventory','logs','results','configs','artifacts/harness-snapshots','artifacts/ray-logs','artifacts/tilelang-wheels']:
    paths.update(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update(p for p in (ROOT/'artifacts').iterdir() if p.is_file())
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
index=ROOT/'EVIDENCE-SHA256.json';index.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(paths|{index}):tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(ROOT/'evidence.tar.gz.sha256').write_text(digest+'  evidence.tar.gz\n')
print(json.dumps(dict(files=len(manifest),bytes=archive.stat().st_size,sha256=digest),indent=2))

```
