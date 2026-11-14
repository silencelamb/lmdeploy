# collect_runtime_logs.py

原始验证文件快照；源码 SHA256：`8fa5db1c64de3c992de3a790f3bd8a9eb3f3c0699a59785db99b2193da144ef5`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Retain full worker logs from the exact Ray sessions owned by each engine."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
index={}
for path in (ROOT/'results').glob('*-external-pr5030*.json'):
    data=json.loads(path.read_text())
    if 'events' not in data or 'options' not in data:continue
    for item in data['events']:
        if item['event']!='ready':continue
        sessions=list(Path('/tmp/ray').glob(f"session_*_{item['pid']}"))
        assert len(sessions)==1,(item,sessions)
        dest=ROOT/'artifacts/ray-logs'/data['options']['label']/item['role']
        dest.mkdir(parents=True,exist_ok=True)
        for source in (sessions[0]/'logs').glob('worker-*'):
            if source.suffix not in ['.out','.err']:continue
            target=dest/source.name;shutil.copy2(source,target)
            index[str(target.relative_to(ROOT))]=dict(source=str(source),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(ROOT/'inventory/ray-log-sources.json').write_text(json.dumps(index,indent=2))
print('Copied',len(index),'owned Ray worker log files')

```
