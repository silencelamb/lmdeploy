"""Rebuild the final upgrade conclusion from immutable run outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/upgrade5-summary-r0.json'
assert not OUT.exists(), OUT

def read(relative): return json.loads((ROOT/relative).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

minimal = []
for name in ('upgrade5-old-mini3-r0','upgrade5-mini3-r0','upgrade5-mini1-r0'):
    d = read('results/'+name+'.json')
    minimal.append(dict(label=name,iterations=len(d['results']),
                        state_bad=sum(x['state_bad']>0 for x in d['results']),
                        max_abs=max(x['max_abs'] for x in d['results'])))
real = read('results/upgrade5-real3-r0.json')
assert len(real['results']) == 200
assert all(x['state_exact_reference'] and x['output_exact_reference'] for x in real['results'])

micro = read('results/upgrade5-full-gdn-perf-r0.json')
assert micro['complete']
micro_rows = []
for case in micro['cases']:
    assert all(v['state_close'] for v in case['correctness'].values())
    s = case['summary']
    micro_rows.append(dict(batch=case['batch'],queries=case['queries'],
        stages3_us=s['stages3_median']*1000,stages1_us=s['stages1_median']*1000,
        stages3_vs_stages1_pct=100*(s['stages3_median']/s['stages1_median']-1)))

target = []
for repeat in (0,1):
    for rank in (0,1):
        old = read(f'results/perf2-target-upgrade5-old1-r{repeat}-rank{rank}.json')
        new = read(f'results/perf2-target-upgrade5-new3-r{repeat}-rank{rank}.json')
        ref = read(f'results/perf2-target-upgrade5-old1-r0-rank{rank}.json')
        keys = ('input_ids','positions','history','state_before_sha256','first_output_sha256')
        matching = {k:old[k]==new[k]==ref[k] for k in keys}
        assert all(matching.values()),matching
        assert len(old['output_hashes'])==len(new['output_hashes'])==48
        assert set(old['output_hashes'])==set(new['output_hashes'])=={ref['first_output_sha256']}
        target.append(dict(repeat=repeat,rank=rank,old_single_ms=old['median_ms'],
                           new_multi_ms=new['median_ms'],
                           new_vs_old_pct=100*(new['median_ms']/old['median_ms']-1),
                           fixed_inputs_and_state=matching,all_48_outputs_identical=True))

logs = []
for p in sorted((ROOT/'logs').glob('upgrade5-*.json')):
    d = json.loads(p.read_text())
    if 'returncode' in d:
        logs.append(dict(label=d['label'],returncode=d['returncode'],
                         source=d.get('source_head'),elapsed_seconds=d.get('elapsed_seconds'),sha256=sha(p)))
assert all(x['returncode']==0 for x in logs), logs
data = dict(original_compiler='1.2.0 / 042c923d02a22e5845cf10311cf2d192b17a5d77',
            new_compiler='2.0.0 / efb7141581c009bbd1d676b18941ed15ede62a68',
            new_full_package='triton-ascend 3.2.2.post2',cann_runtime='9.1.0 unchanged',
            wheel_sha256='ca16e1f6df5850663e430789e1d68462d902bb4701a6f0441c15e175bb132005',
            minimal=minimal,real_fixture=dict(iterations=200,state_exact=200,output_exact=200),
            hardware=dict(component_only='6 passed',full_package='6 passed'),
            mtp=read('results/upgrade5-mtp-comparison-r0.json'),micro=micro_rows,target=target,logs=logs,
            limitations=['CANN9.2 runtime and driver were not upgraded or tested.',
                         'No HTTP throughput / TTFT / TPOT conclusion.',
                         'CPU18 callback still requires privileged host capture.',
                         'A complete v2.0.0 release was tested, not an isolated 05f2adcf cherry-pick.'])
OUT.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(micro=micro_rows,target=target),indent=2))
