"""Check the complete constrained app stack in an independently owned venv."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.parse

from pip._vendor.cachecontrol.controller import CacheController

ROOT=Path(__file__).resolve().parents[1]
HOUSE=ROOT/'artifacts/ci8-wheelhouse'
HOUSE.mkdir(exist_ok=False)
report=json.loads((ROOT/'results/ci8-dependency-resolve-r0.json').read_text())
records=[]
for item in report['install']:
    info=item['download_info'];url=info['url']
    if not urllib.parse.urlparse(url).path.endswith('.whl'):continue
    key=hashlib.sha224(CacheController.cache_url(url).encode()).hexdigest()
    cached=Path('/root/.cache/pip/http-v2').joinpath(*list(key[:5]),key+'.body')
    if not cached.exists():continue
    expected=info.get('archive_info',{}).get('hashes',{}).get('sha256')
    with cached.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert expected==actual,(url,expected,actual)
    name=urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit('/',1)[1])
    os.link(cached,HOUSE/name)
    records.append(dict(file=name,url=url,sha256=actual,bytes=cached.stat().st_size))
app=next((ROOT/'artifacts/ci8-dist-r1').glob('*.whl'))
shutil.copyfile(app,HOUSE/app.name)
(ROOT/'inventory/ci8-wheelhouse-r0.json').write_text(json.dumps(records,indent=2)+'\n')
VENV=ROOT/'envs/ci8-clean'
assert not VENV.exists()
subprocess.run(['/usr/local/python3.12.13/bin/python3.12','-m','venv','--copies',str(VENV)],check=True)
special=[next(HOUSE.glob(p)) for p in ('torch-2.10.0+cpu-*.whl','torchvision-0.25.0+cpu-*.whl','triton_ascend-3.2.2.post2-*.whl')]
cmd=[str(ROOT/'tools/ci8-python-tools/bin/uv'),'pip','install','--python',str(VENV/'bin/python'),
     '--index-url','https://pypi.org/simple','--find-links',str(HOUSE),
     '-r',str(ROOT/'results/ci8-resolved-requirements-r5.txt'),str(HOUSE/app.name),*map(str,special)]
env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONNOUSERSITE='1',
         PYTHONPATH=str(ROOT/'guard-ci8'),LMDEPLOY_TARGET_DEVICE='ascend')
record=dict(command=cmd,venv=str(VENV),wheelhouse_files=len(records))
print('INSTALL',json.dumps(record),flush=True)
record['returncode']=subprocess.run(cmd,env=env,cwd=ROOT/'lmdeploy-ci8').returncode
if record['returncode']==0:
    # Triton-Ascend officially overlays the triton namespace. Ensure the vendor
    # wheel is last, as with pip's dependency-before-dependent install ordering.
    vendor=next(HOUSE.glob('triton_ascend-3.2.2.post2-*.whl'))
    final=[str(VENV/'bin/python'),'-m','pip','install','--no-deps','--no-index','--force-reinstall',str(vendor)]
    record['vendor_overlay_command']=final
    record['vendor_overlay_returncode']=subprocess.run(final,env=env,cwd=ROOT/'lmdeploy-ci8').returncode
    record['pip_check_returncode']=subprocess.run([str(VENV/'bin/python'),'-m','pip','check'],env=env,cwd=ROOT/'lmdeploy-ci8').returncode
(ROOT/'logs/ci8-fresh-install-r0.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(record['returncode'] or record.get('vendor_overlay_returncode',0) or record.get('pip_check_returncode',0))
