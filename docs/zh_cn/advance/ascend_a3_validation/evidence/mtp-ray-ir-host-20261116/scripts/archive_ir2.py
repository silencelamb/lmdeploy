"""Publish new IR/host/performance evidence without touching older archives."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/mtp-ray-ir-host-20261116')
DEST.mkdir(parents=True,exist_ok=False)
prefixes=('ir2-','host2-','perf2-')
files=[]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,rel,compressed=False):
    dest=DEST/rel
    if compressed:dest=dest.with_name(dest.name+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if compressed:
        with p.open('rb') as src,dest.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:shutil.copyfileobj(src,out)
    else:shutil.copyfile(p,dest)
    files.append(dict(path=str(dest.relative_to(DEST)),source=str(p),bytes=dest.stat().st_size,
        sha256=sha(dest),source_bytes=p.stat().st_size,source_sha256=sha(p),compressed=compressed))
for folder in ('logs','results','inventory'):
    for p in sorted((ROOT/folder).glob('*')):
        if p.is_file() and p.name.startswith(prefixes) and p.suffix!='.pt':
            copy(p,p.relative_to(ROOT),compressed=p.stat().st_size>1_000_000)
copy(ROOT/'ACCEPTANCE-ir2.md',Path('ACCEPTANCE-ir2.md'))
for p in sorted((ROOT/'scripts').glob('*.py')):
    if p.name.startswith(('gdn_ir2','ir2_','perf2_','host2_','summarize_ir2','archive_ir2')):
        copy(p,Path('scripts')/p.name)
for p in sorted((ROOT/'artifacts').glob('*.patch')):
    if p.name.startswith(prefixes):copy(p,Path('artifacts')/p.name)
for p in sorted((ROOT/'artifacts').glob('host2-*/preflight.json')):
    copy(p,Path('artifacts')/p.parent.name/p.name)

with tarfile.open(DEST/'host-tools.tar.gz','w:gz') as tar:
    for p in sorted((ROOT/'lmdeploy-research/tools/ascend_a3/diagnostics').glob('*')):
        if p.is_file():tar.add(p,arcname='tools/ascend_a3/diagnostics/'+p.name)

ir_files=[]
for p in sorted((ROOT/'artifacts').glob('ir2-*')):
    if p.is_file() and p.suffix in ('.mlir','.patch'):ir_files.append(p)
    if not p.is_dir() or p.name.endswith('-scripts') or 'source-fixed' in p.name or 'debug-dumps' in p.name:continue
    for f in p.rglob('*'):
        if f.is_file() and 'triton-cache' not in f.parts and f.suffix not in ('.pt',):ir_files.append(f)
with tarfile.open(DEST/'ir-artifacts.tar.gz','w:gz') as tar:
    for p in ir_files:tar.add(p,arcname=str(p.relative_to(ROOT)))
with tarfile.open(DEST/'script-snapshots.tar.gz','w:gz') as tar:
    for p in sorted((ROOT/'artifacts').glob('*-scripts')):
        if p.name.startswith(prefixes):tar.add(p,arcname=p.name)

local=[]
for p in sorted((ROOT/'artifacts').glob('ir2-*/*.pt')):
    local.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
local.append(dict(path=str(ROOT/'results/gdn-real-fixed-graph-r0.pt'),bytes=(ROOT/'results/gdn-real-fixed-graph-r0.pt').stat().st_size,sha256=sha(ROOT/'results/gdn-real-fixed-graph-r0.pt')))
(DEST/'local-tensors.json').write_text(json.dumps(dict(files=local,note='Tensor fixtures remain on A3; standalone synthetic reproducer requires no model files.'),indent=2)+'\n')
bundles=[]
for name in ('host-tools.tar.gz','ir-artifacts.tar.gz','script-snapshots.tar.gz','local-tensors.json'):
    p=DEST/name;bundles.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
index=dict(native='5f1b1b3c12b1c198813f65275f778c7ad292fb02',fixed_execution='d2d933dad7c6eb55b26ad5b83667c2077e23ca6c',
    research_head=subprocess.check_output(['git','-C',str(ROOT/'lmdeploy-research'),'rev-parse','HEAD'],text=True).strip(),
    compiler_commit='042c923d02a22e5845cf10311cf2d192b17a5d77',files=files,bundles=bundles,
    limits=['Host preflight only; actual tasklet tracing not executed.','Compiler source candidate not built.','Invalid frozen-target diagnostic runs retained and excluded in summary.'])
(DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(dict(files=len(files)+len(bundles),bytes=sum(x['bytes'] for x in files+bundles),local_tensors=len(local)),indent=2))
