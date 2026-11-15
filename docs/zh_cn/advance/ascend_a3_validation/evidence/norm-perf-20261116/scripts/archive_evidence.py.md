# archive_evidence.py

```python
"""Archive this investigation, including failures, only on the fork docs branch."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/norm-perf-20261116')
DEST.mkdir(parents=True,exist_ok=True)
records=[]


def copy(source, relative, markdown=False):
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if markdown:
        target.write_text(f'# {source.name}\n\n```python\n{source.read_text().rstrip()}\n```\n')
    else:
        shutil.copyfile(source,target)
    records.append(dict(path=str(relative),source=str(source),bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))


for directory in ('logs','results','inventory'):
    for source in sorted((ROOT/directory).rglob('*')):
        if source.is_file():
            copy(source,source.relative_to(ROOT))
for directory in ('scripts','guard'):
    for source in sorted((ROOT/directory).glob('*.py')):
        copy(source,Path(directory)/(source.name+'.md'),markdown=True)
for source in sorted((ROOT/'artifacts').glob('*.patch')):
    copy(source,Path('patches')/source.name)
copy(ROOT/'ACCEPTANCE.md',Path('ACCEPTANCE.md'))
index=dict(local_root=str(ROOT),source_commits=json.loads((ROOT/'inventory/completion.json').read_text())['sources'],
    files=records,excluded='Native environment, py-spy wheel/binary, entire Triton cache and temporary progress summaries. Actual kernel metadata/IR/hashes are included.')
(DEST/'index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records)),indent=2))
```
