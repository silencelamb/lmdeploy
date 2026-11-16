"""Archive PR integration evidence without changing earlier investigation data."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/mtp-pr-integration-20261116')
DEST.mkdir(parents=True,exist_ok=False)
records=[]

def copy(path,relative=None):
    raw=path.read_bytes()
    data=re.sub(rb'("(?:Authorization|Cookie|Set-Cookie)"\s*:\s*)"(?:\\.|[^"\\])*"',
                rb'\1"[redacted]"',raw,flags=re.I)
    out=DEST/(relative or path.relative_to(ROOT));out.parent.mkdir(parents=True,exist_ok=True)
    compressed=len(data)>1_000_000
    if compressed:
        out=out.with_name(out.name+'.gz')
        with out.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as stream:stream.write(data)
    else:out.write_bytes(data)
    records.append(dict(path=str(out.relative_to(DEST)),source=str(path),bytes=out.stat().st_size,
                        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(raw).hexdigest(),compressed=compressed))

for folder in ('logs','results','inventory'):
    for path in sorted((ROOT/folder).iterdir()):
        if path.is_file() and 'pr6' in path.name and path.name!='pr6-archive-r0.log':copy(path)
for path in (ROOT/'ACCEPTANCE-pr6.md',ROOT/'HANDOFF-pr6.md',ROOT/'artifacts/pr6-pr-body.md'):copy(path)
for path in sorted((ROOT/'scripts').glob('*.py')):
    if path.name.startswith('pr6_') or path.name in ('run_upgrade5.py','upgrade5_inventory.py','mtp_deep_trace.py','mtp_audit.py','pipeline_case.py'):copy(path)
copy(ROOT/'guard/sitecustomize.py')
for path in sorted((ROOT/'artifacts').glob('pr6-*.patch')):copy(path)
for path in sorted((ROOT/'artifacts').glob('mtp-deep-pr6-*/*/seed.json')):copy(path)
with tarfile.open(DEST/'script-snapshots.tar.gz','w:gz') as archive:
    for path in sorted((ROOT/'artifacts').glob('pr6-*-scripts')):archive.add(path,arcname=path.name)
p=DEST/'script-snapshots.tar.gz'
records.append(dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(DEST/'index.json').write_text(json.dumps(dict(source_head='85a0581228ad33a695430a1f43a47358a8258131',
    previous_pr_head='5f1b1b3c12b1c198813f65275f778c7ad292fb02',
    pr='https://github.com/InternLM/lmdeploy/pull/5134',files=records,
    limitations=['No new performance or full migration/Docker certification in this integration.',
                 'CPU18 host callback/permanent fix remains unresolved; no affinity or RPC-order workaround shipped.']),indent=2)+'\n')
print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),destination=str(DEST)),indent=2))
