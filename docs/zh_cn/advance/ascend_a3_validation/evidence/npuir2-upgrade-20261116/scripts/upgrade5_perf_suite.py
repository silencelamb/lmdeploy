"""Sequential performance checks after the upgrade correctness gates."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'scripts/run_upgrade5.py'
NEW = ROOT / 'envs/upgrade5-full322p2'
OLD = Path('/root/workspace/a3-mooncake-pr5030-20261113/envs/native')
for name in ('upgrade5-mtp-r0','upgrade5-mtp-r1','upgrade5-full-mtp-r0','upgrade5-full-mtp-r1'):
    d = json.loads((ROOT / 'results' / (name + '.json')).read_text())
    assert d['complete'] and not d['dlinfer_modules'], name
    assert json.loads((ROOT / 'logs' / (name + '.json')).read_text())['returncode'] == 0
    reference = json.loads((ROOT / 'results/mtp-stage1-suite-r0.json').read_text())
    assert len(d['events']) == len(reference['events'])
    for event, old in zip(d['events'], reference['events']):
        assert len(event['responses']) == len(old['responses'])
        for x,y in zip(event['responses'],old['responses']):
            assert x['token_ids'] == y['token_ids'] and x['logprobs'] == y['logprobs'], name

def run(label, venv, repo, cache, args, perf_label=None):
    cmd = [sys.executable,str(RUN),'--label',label,'--venv',str(venv),'--repo',str(repo),
           '--compiler-cache',cache,'--devices','0,1','--timeout','900','--',*map(str,args)]
    env = dict(os.environ, PYTHONHASHSEED='42')
    if perf_label: env['PERF2_LABEL'] = perf_label
    print('START',label,flush=True)
    result = subprocess.run(cmd,env=env)
    print('END',label,result.returncode,flush=True)
    if result.returncode: raise SystemExit(result.returncode)

run('upgrade5-full-gdn-perf-r0',NEW,ROOT/'lmdeploy-upgrade5','upgrade5-full322p2',
    [ROOT/'scripts/upgrade5_perf_gdn.py','--rounds','8','--iterations','500',
     '--output',ROOT/'results/upgrade5-full-gdn-perf-r0.json'])

for variant, repeat in (('old1',0),('new3',0),('new3',1),('old1',1)):
    old = variant == 'old1'
    label = f'upgrade5-target-{variant}-r{repeat}'
    run(label,OLD if old else NEW,ROOT/('lmdeploy-research' if old else 'lmdeploy-upgrade5'),
        'upgrade5-old' if old else 'upgrade5-full322p2',
        [ROOT/'scripts/perf2_frozen_target_v2.py','--model','/afs-weights/Sage-A1.5',
         '--tp','2','--mtp','3','--prefix','--suite','smoke','--new-tokens','8',
         '--cache-fraction','0.7','--output',ROOT/'results'/f'{label}-pipeline.json'],
        perf_label=f'upgrade5-{variant}-r{repeat}')
