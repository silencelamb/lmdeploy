# preservation_audit.py

原始验证文件快照；源码 SHA256：`f2c0339c921dec274f5cc691854e8388b0e8485c2cee786ec09a2db5da764e66`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Read-only audit against the earlier snapshots and this follow-up's references."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/workspace/a3-validation-20261112')
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks={})
for name,path in [('lmdeploy','/root/workspace/lmdeploy'),('dlinfer','/root/workspace/dlinfer'),
                  ('code-lmdeploy','/root/workspace/code/lmdeploy'),('code-dlinfer','/root/workspace/code/dlinfer'),
                  ('code-triton-ascend-kernels','/root/workspace/code/triton-ascend-kernels')]:
    before=json.loads((OLD/'inventory'/f'{name}.json').read_text())
    result['checks'][name]=dict(head_unchanged=git(path,'rev-parse','HEAD').strip()==before['head'].strip(),
        diff_unchanged=git(path,'diff','--binary')==before['diff'])
before=json.loads((ROOT/'inventory/colleague-lmdeploy.json').read_text())
path='/afs-weights/wangqing/kv_offloading_dev/code/lmdeploy'
result['checks']['wangqing-lmdeploy']=dict(head_unchanged=git(path,'rev-parse','HEAD')==before['head'],
    diff_unchanged=git(path,'diff','--binary')==before['diff'])
for path,want in json.loads((ROOT/'inventory/reference-files.json').read_text()).items():
    result['checks'][path]=dict(sha256_unchanged=hashlib.sha256(Path(path).read_bytes()).hexdigest()==want)
for name,expected in json.loads((ROOT/'inventory/comparison-revisions.json').read_text()).items():
    path=ROOT/'sources'/f'lmdeploy-{name}'
    result['checks'][f'comparison-{name}']=dict(expected_head=git(path,'rev-parse','HEAD').strip()==expected['head'],
        clean=not git(path,'status','--porcelain'))
for manifest in (OLD/'inventory').glob('model-*.json'):
    model=json.loads(manifest.read_text());changed=[]
    for rel,stat in model['files'].items():
        current=(Path(model['path'])/rel).stat()
        if current.st_size!=stat['bytes'] or current.st_mtime_ns!=stat['mtime_ns']:changed.append(rel)
    result['checks'][model['path']]=dict(size_mtime_unchanged=not changed)
archive=OLD/'evidence.tar.gz'
expected=(OLD/'evidence.tar.gz.sha256').read_text().split()[0]
result['checks']['original-evidence-archive']=dict(sha256_unchanged=hashlib.sha256(archive.read_bytes()).hexdigest()==expected)
forbidden=ROOT/'logs/forbidden-dlinfer-imports.jsonl'
events=forbidden.read_text().splitlines() if forbidden.exists() else []
site=ROOT/'envs/native/lib/python3.12/site-packages'
result['checks']['native-isolation']=dict(no_forbidden_import_events=not events,
    no_dlinfer_artifacts=not any('dlinfer' in p.name.lower() for p in site.iterdir()),
    guard_present='sys.meta_path.insert(0, NoDlinfer())' in (site/'sitecustomize.py').read_text())
result['pass']=all(all(row.values()) for row in result['checks'].values())
(ROOT/'inventory/preservation-final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2));assert result['pass']

```
