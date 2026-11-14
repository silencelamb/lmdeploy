# model_manifest.py

原始验证文件快照；源码 SHA256：`11ec92ef3c321ab3717a27922dc2c695df615771205d12b228e91fbe98dcab92`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import hashlib
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
for name in (sys.argv[1:] or ['Sage-A1.5','Qwen3-8B','Qwen3.5-0.8B']):
    model=Path('/afs-weights')/name
    files={}
    for f in sorted(model.iterdir()):
        if f.is_file() and f.suffix in ['.json','.jinja','.safetensors']:
            stat=f.stat()
            with f.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files[f.name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest)
            print(name,f.name,flush=True)
    (root/'inventory'/f'model-{name}.json').write_text(json.dumps(dict(path=str(model),files=files),indent=2))

```
