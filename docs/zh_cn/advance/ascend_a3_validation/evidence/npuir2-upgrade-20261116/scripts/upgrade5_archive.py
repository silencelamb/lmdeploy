"""Publish bounded upgrade evidence; retain packages/tensors locally."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/ascend_a3_validation/evidence/npuir2-upgrade-20261116')
DEST.mkdir(parents=True, exist_ok=False)
records = []

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def publish(path, relative=None):
    relative = relative or path.relative_to(ROOT)
    raw = path.read_bytes()
    # Public-download WAF tracking cookies are irrelevant to reproduction.
    public = re.sub(rb'("(?:Set-Cookie|Authorization|Cookie)"\s*:\s*)"(?:\\.|[^"\\])*"',
                    rb'\1"[redacted]"',raw,flags=re.IGNORECASE)
    compressed = len(public) > 750_000
    dest = DEST / relative
    if compressed: dest = dest.with_name(dest.name+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if compressed:
        with dest.open('wb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as output: output.write(public)
    else: dest.write_bytes(public)
    records.append(dict(path=str(dest.relative_to(DEST)),source=str(path),bytes=dest.stat().st_size,
                        sha256=digest(dest),source_sha256=hashlib.sha256(raw).hexdigest(),
                        compressed=compressed,tracking_headers_redacted=public!=raw))

for folder in ('logs','results','inventory'):
    for path in sorted((ROOT/folder).iterdir()):
        if path.is_file() and ('upgrade5' in path.name) and path.suffix != '.pt': publish(path)
for path in (ROOT/'ACCEPTANCE-upgrade5.md',ROOT/'HANDOFF-upgrade5.md'): publish(path)
for path in sorted((ROOT/'scripts').glob('*.py')):
    if path.name.startswith(('upgrade5_','run_upgrade5')) or path.name in (
        'gdn_ir2_probe.py','run_gdn_pytest_pinned.py','perf2_frozen_target_v2.py',
        'mtp_deep_trace.py','mtp_audit.py','pipeline_case.py'):
        publish(path)
publish(ROOT/'guard/sitecustomize.py')
publish(ROOT/'lmdeploy-research/tools/ascend_a3/diagnostics/reproduce_state_writeback.py',
        Path('standalone/reproduce_state_writeback.py'))
publish(ROOT/'lmdeploy-research/tests/pytorch/ascend/hardware/test_gdn_state_writeback.py',
        Path('hardware/test_gdn_state_writeback.py'))
for path in sorted((ROOT/'artifacts/upgrade5-real3-r0').rglob('*')):
    if path.is_file() and 'triton-cache' not in path.parts and path.suffix != '.pt':
        publish(path)

snapshots = DEST/'script-snapshots.tar.gz'
with tarfile.open(snapshots,'w:gz') as archive:
    for path in sorted((ROOT/'artifacts').glob('upgrade5-*-scripts')):
        archive.add(path,arcname=path.name)
records.append(dict(path=snapshots.name,bytes=snapshots.stat().st_size,sha256=digest(snapshots)))
local = []
for pattern in ('artifacts/upgrade5-downloads-r1/*.whl','artifacts/upgrade5-downloads-r1/*.run',
                'artifacts/upgrade5-real3-r0/*.pt','results/gdn-real-fixed-graph-r0.pt'):
    for path in ROOT.glob(pattern):
        local.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
(DEST/'index.json').write_text(json.dumps(dict(
    native_head='5f1b1b3c12b1c198813f65275f778c7ad292fb02',
    old_correct_head='f9ab50d877c384bfb04ae43afa6af0c388fc451b',
    original_compiler='042c923d02a22e5845cf10311cf2d192b17a5d77',
    upgrade_compiler='efb7141581c009bbd1d676b18941ed15ede62a68',
    files=records,local_only=local,
    limits=['No full CANN9.2 runtime or driver upgrade.',
            'No end-to-end HTTP throughput validation.',
            'CPU18 excluded for benchmark processes; host callback identity still pending.']
),indent=2)+'\n')
print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),destination=str(DEST)),indent=2))
