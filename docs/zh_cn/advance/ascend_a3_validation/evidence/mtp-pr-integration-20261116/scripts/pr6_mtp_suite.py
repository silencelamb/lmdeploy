"""Two fresh MTP runs per compiler profile on the proposed migration PR head."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'lmdeploy-pr6'
PROFILES = {
    'old': (Path('/root/workspace/a3-mooncake-pr5030-20261113/envs/native'), '0,1'),
    'new': (ROOT / 'envs/upgrade5-full322p2', '2,3'),
}

for label in ('pr6-old-hw-r0','pr6-new-hw-r0','pr6-cpu-r0'):
    assert json.loads((ROOT/'logs'/f'{label}.json').read_text())['returncode'] == 0

def run_profile(profile):
    venv, devices = PROFILES[profile]
    for repeat in (0,1):
        label = f'pr6-{profile}-mtp-r{repeat}'
        env = dict(os.environ, PYTHONHASHSEED='42', VALIDATION_AUDIT_LABEL=label,
                   MTP_DEEP_TRACE='none')
        command = [sys.executable,str(ROOT/'scripts/run_upgrade5.py'),
                   '--repo',str(REPO),'--venv',str(venv),'--compiler-cache',f'pr6-{profile}',
                   '--label',label,'--devices',devices,'--timeout','900','--',
                   str(ROOT/'scripts/mtp_deep_trace.py'),'--model','/afs-weights/Sage-A1.5',
                   '--tp','2','--mtp','3','--prefix','--suite','trace','--scores',
                   '--new-tokens','64','--cache-fraction','0.95',
                   '--output',str(ROOT/'results'/f'{label}.json')]
        print('START',label,flush=True)
        with (ROOT/'logs'/f'{label}-launcher.log').open('x') as output:
            rc = subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
        print('END',label,rc,flush=True)
        if rc: raise RuntimeError((label,rc))

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run_profile,PROFILES))
