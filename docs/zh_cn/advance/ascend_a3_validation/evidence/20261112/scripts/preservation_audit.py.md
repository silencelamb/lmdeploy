# preservation_audit.py

原始验证文件快照；源码 SHA256：`ace394783c5a053fba6c342058d4f67869f5f227a4bfe7930159ecbabb1a6b44`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Check preserved workspaces, fixed sources, model metadata and import isolation."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True)

result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              original_workspaces={}, comparison_worktrees={}, models={}, external_kernel={})
for label, path in [('lmdeploy', WORK/'lmdeploy'), ('dlinfer', WORK/'dlinfer'),
                    ('code-lmdeploy', WORK/'code/lmdeploy'),
                    ('code-dlinfer', WORK/'code/dlinfer'),
                    ('code-triton-ascend-kernels', WORK/'code/triton-ascend-kernels')]:
    before=json.loads((ROOT/'inventory'/f'{label}.json').read_text())
    head=git(path, 'rev-parse', 'HEAD')
    diff=git(path, 'diff', '--binary')
    result['original_workspaces'][label]=dict(head=head.strip(),
        head_unchanged=head.strip()==before['head'].strip(), diff_unchanged=diff==before['diff'],
        status=git(path, 'status', '--short', '--branch'))
for label, expected in json.loads((ROOT/'inventory/comparison-revisions.json').read_text()).items():
    path=ROOT/'sources'/label
    head=git(path, 'rev-parse', 'HEAD').strip()
    status=git(path, 'status', '--porcelain')
    result['comparison_worktrees'][label]=dict(head=head, expected_head=head==expected['head'],
                                               clean=not status, status=status)
for manifest in (ROOT/'inventory').glob('model-*.json'):
    data=json.loads(manifest.read_text())
    mismatches=[]
    for rel, expected in data['files'].items():
        path=Path(data['path'])/rel
        stat=path.stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
            mismatches.append(rel)
    result['models'][data['path']]=dict(files=len(data['files']),
        size_and_mtime_unchanged=not mismatches, mismatches=mismatches,
        hash_note='Initial full SHA256 retained; final check compares size and mtime, not rehashed weights.')
manifest=json.loads((ROOT/'inventory/external-kernel-files.sha256.json').read_text())
for label, path in [('original', WORK/'code/triton-ascend-kernels'),
                    ('snapshot', ROOT/'sources/triton-ascend-kernels-snapshot')]:
    mismatches=[rel for rel, digest in manifest.items()
        if hashlib.sha256((path/rel).read_bytes()).hexdigest()!=digest]
    result['external_kernel'][label]=dict(files=len(manifest), sha256_unchanged=not mismatches,
                                        mismatches=mismatches)
result['colleague_install_script_unchanged']=(WORK/'code/install.sh').read_bytes()==(ROOT/'inventory/install.sh').read_bytes()
forbidden=ROOT/'logs/forbidden-dlinfer-imports.jsonl'
result['native_forbidden_import_events']=len(forbidden.read_text().splitlines()) if forbidden.exists() else 0
site=ROOT/'envs/native/lib/python3.12/site-packages'
result['native_dlinfer_artifacts']=[p.name for p in site.iterdir() if 'dlinfer' in p.name.lower()]
result['native_guard_present']='sys.meta_path.insert(0, NoDlinfer())' in (site/'sitecustomize.py').read_text()
result['mooncake_worktrees']={}
mc_revisions=ROOT/'inventory/mooncake-comparison-revisions.json'
if mc_revisions.exists():
    for env, expected in json.loads(mc_revisions.read_text()).items():
        path=ROOT/'sources'/f'lmdeploy-{env}-mooncake'
        head=git(path,'rev-parse','HEAD').strip();status=git(path,'status','--porcelain')
        result['mooncake_worktrees'][env]=dict(head=head,expected_head=head==expected['head'],clean=not status,status=status)
mc_site=ROOT/'envs/native-mooncake/lib/python3.12/site-packages'
if mc_site.exists():
    result['native_mooncake_dlinfer_artifacts']=[p.name for p in mc_site.iterdir() if 'dlinfer' in p.name.lower()]
    result['native_mooncake_guard_present']='sys.meta_path.insert(0, NoDlinfer())' in (mc_site/'sitecustomize.py').read_text()
result['pass']=(all(x['head_unchanged'] and x['diff_unchanged'] for x in result['original_workspaces'].values())
    and all(x['expected_head'] and x['clean'] for x in result['comparison_worktrees'].values())
    and all(x['size_and_mtime_unchanged'] for x in result['models'].values())
    and all(x['sha256_unchanged'] for x in result['external_kernel'].values())
    and result['colleague_install_script_unchanged'] and not result['native_forbidden_import_events']
    and not result['native_dlinfer_artifacts'] and result['native_guard_present']
    and all(x['expected_head'] and x['clean'] for x in result['mooncake_worktrees'].values())
    and not result.get('native_mooncake_dlinfer_artifacts',[]) and result.get('native_mooncake_guard_present',True))
(ROOT/'inventory/preservation-final.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result['pass']

```
