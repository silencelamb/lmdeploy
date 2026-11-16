"""Archive the CANN9.1 and CI investigation without copying model tensors."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/ci-docker-cann91-20261116')
DEST.mkdir(parents=True,exist_ok=False)
rows=[]
def publish(p,rel=None):
    raw=p.read_bytes()
    data=re.sub(rb'("(?:Authorization|Cookie|Set-Cookie)"\s*:\s*)"(?:\\.|[^"\\])*"',rb'\1"[redacted]"',raw,flags=re.I)
    data=re.sub(rb'(?im)^Set-Cookie:.*$',b'Set-Cookie: [redacted]',data)
    out=DEST/(rel or p.relative_to(ROOT));out.parent.mkdir(parents=True,exist_ok=True)
    zipped=len(data)>750_000
    if zipped:
        out=out.with_name(out.name+'.gz')
        with out.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
    else:out.write_bytes(data)
    rows.append(dict(path=str(out.relative_to(DEST)),source=str(p),bytes=out.stat().st_size,
                     sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(raw).hexdigest(),compressed=zipped))
for folder in ('logs','results','inventory'):
    for p in sorted((ROOT/folder).iterdir()):
        if p.is_file() and 'ci8' in p.name and p.name!='ci8-archive-r0.log':publish(p)
for p in (ROOT/'ACCEPTANCE-ci8.md',ROOT/'HANDOFF-ci8.md',ROOT/'artifacts/ci8-gpu-validation-prompt.txt',ROOT/'artifacts/ci8-pr-body-r0.md'):publish(p)
for p in sorted((ROOT/'scripts').glob('*.py')):
    if p.name.startswith(('ci8_','run_ci8')) or p.name in ('archive_ci8.py','run_upgrade5.py','mtp_deep_trace.py','mtp_audit.py','pipeline_case.py','pr6_run_hardware.py'):publish(p)
publish(ROOT/'guard-ci8/sitecustomize.py')
for p in sorted((ROOT/'artifacts').glob('ci8-*.patch')):publish(p)
for p in sorted((ROOT/'artifacts').glob('mtp-deep-ci8-*/*/seed.json')):publish(p)
with tarfile.open(DEST/'script-snapshots.tar.gz','w:gz') as z:
    for p in sorted((ROOT/'artifacts').glob('ci8-*-scripts')):z.add(p,arcname=p.name)
source=Path('/root/workspace/a3-validation-20261112/sources/triton-ascend-kernels-snapshot')
with tarfile.open(DEST/'external-kernel-reference.tar.gz','w:gz') as z:
    for p in sorted(source.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
            z.add(p,arcname=str(p.relative_to(source)))
for name in ('script-snapshots.tar.gz','external-kernel-reference.tar.gz'):
    p=DEST/name;rows.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(DEST/'index.json').write_text(json.dumps(dict(head='111dbf6b4dc1c1aa9fb2fd0753313e0314858c96',baseline='442097e83799c2e85df9d6bbce35c7c79ea05dfe',
    files=rows,ci_state='new-head checks running at initial publication',limits=['CANN9.2 cancelled; no runtime/driver upgrade.','Jetson inherited ABI issue kept separate.','Full six-model CUDA validation pending.']),indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows)),indent=2))
