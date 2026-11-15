# archive_deep.py

```python
"""Archive the continuation without altering the earlier evidence directory."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/mtp-ray-deep-20261116')
CUTOFF=datetime.datetime(2026,11,15,17,31,tzinfo=datetime.timezone.utc).timestamp()
DEST.mkdir(parents=True,exist_ok=False)
entries=[]

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy(p,rel,compress=False,markdown=False):
    target=DEST/rel
    if compress:target=target.with_name(target.name+'.gz')
    if markdown:target=target.with_name(target.name+'.md')
    target.parent.mkdir(parents=True,exist_ok=True)
    if markdown:
        language='cpp' if p.suffix=='.cc' else 'python'
        target.write_text(f'# {p.name}\n\n```{language}\n{p.read_text().rstrip()}\n```\n')
    elif compress:
        with p.open('rb') as src,target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:
            shutil.copyfileobj(src,out)
    else:shutil.copyfile(p,target)
    entries.append(dict(path=str(target.relative_to(DEST)),source=str(p),bytes=target.stat().st_size,
        sha256=digest(target),source_bytes=p.stat().st_size,source_sha256=digest(p),compressed=compress))

for directory in ['results','logs','inventory']:
    for p in sorted((ROOT/directory).glob('*')):
        if not p.is_file() or p.stat().st_mtime<CUTOFF or p.suffix=='.pt':continue
        if p.name.startswith('archive-deep-') or p.name=='deep-publish-status-r1.json':continue
        if directory=='inventory' and p.name.startswith('ray-source-'):continue
        compress=directory=='results' and p.stat().st_size>2_000_000 and not any(t in p.name for t in ['summary','compare','correlation'])
        copy(p,p.relative_to(ROOT),compress=compress)
for directory in ['scripts','guard']:
    for p in sorted((ROOT/directory).iterdir()):
        if p.is_file() and p.suffix in ['.py','.cc']:copy(p,p.relative_to(ROOT),markdown=True)
for p in sorted((ROOT/'artifacts').glob('*.patch')):
    if p.stat().st_mtime>=CUTOFF:copy(p,Path('patches')/p.name)
for p in sorted((ROOT/'artifacts').glob('ray-*/*')):
    if not p.is_file() or p.stat().st_mtime<CUTOFF:continue
    if p.name.startswith('strace.'):continue
    if p.name.endswith('.log') or p.name.endswith('.json') or p.name=='sched.jsonl':
        copy(p,Path('runtime')/p.parent.name/p.name,compress=p.stat().st_size>400_000)
for p in [ROOT/'ACCEPTANCE-deep-r1.md',ROOT/'artifacts/mtp_deep_trace-seed-r0-failed.py',ROOT/'artifacts/pipe_control-r0-source.cc',ROOT/'artifacts/test_gdn_state_writeback-initial.py']:
    copy(p,Path(p.name),markdown=p.suffix in ['.py','.cc'])
with tarfile.open(DEST/'scripts.tar.gz','w:gz') as tar:
    for directory in ['scripts','guard']:
        for p in sorted((ROOT/directory).iterdir()):
            if p.is_file() and p.suffix in ['.py','.cc']:tar.add(p,arcname=str(p.relative_to(ROOT)))
with tarfile.open(DEST/'script-snapshots.tar.gz','w:gz') as tar:
    for p in sorted((ROOT/'artifacts').glob('*-scripts')):tar.add(p,arcname=p.name,recursive=True)
local=[]
for pattern in ['artifacts/mtp-deep-*/*/*.pt','results/*.pt','artifacts/ray-*/strace.*','artifacts/ray-*-strace.*','artifacts/triton-cache/*/fused_recurrent_gated_delta_rule_fwd_kernel.*']:
    for p in sorted(ROOT.glob(pattern)):
        if p.is_file() and p.stat().st_mtime>=CUTOFF:
            local.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
(DEST/'local-artifacts-manifest.json').write_text(json.dumps(dict(files=local,
    note='These raw tensors, syscall traces and compiler binaries remain on the A3; they are not GitHub attachments.'),indent=2)+'\n')
extra=[]
for name in ['scripts.tar.gz','script-snapshots.tar.gz','local-artifacts-manifest.json']:
    p=DEST/name;extra.append(dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
index=dict(local_root=str(ROOT),original_source='5f1b1b3c12b1c198813f65275f778c7ad292fb02',
    research_head=subprocess.check_output(['git','-C',str(ROOT/'lmdeploy-research'),'rev-parse','HEAD'],text=True).strip(),
    fix_commit='dfa3049f1575305b7564d6f15c48022661bd902a',files=entries,bundles=extra,
    scope='Continuation only. Earlier mtp-ray-20261116 evidence is unchanged. Gzipped files decompress to source bytes with the recorded SHA256; readable scripts additionally carry their source hash.')
(DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(dict(files=len(entries),published_bytes=sum(x['bytes'] for x in entries+extra),local_files=len(local),local_bytes=sum(x['bytes'] for x in local)),indent=2))
```
