"""Summarize this continuation without overwriting prior evidence."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'results'/name).read_text())
out=ROOT/'results/ir2-host2-summary-r0.json';assert not out.exists(),out
cases={}
for label in ['ir2-multi-r1','ir2-single-r1','ir2-binary-control-r0','ir2-binary-fence-r0','ir2-binary-selector-r0']:
    d=read(label+'.json');rs=d['results']
    cases[label]=dict(iterations=len(rs),state_bad=sum(not r['state_close_reference'] for r in rs),
        output_bad=sum(not r['output_exact_reference'] for r in rs),max_abs=max(r['max_abs'] for r in rs),
        input_hashes=d['input_hashes'])
base=ROOT/'artifacts/ir2-multi-r1/kernel.npubin'
control=ROOT/'artifacts/ir2-hivmc-control-r0/kernel.o'
assert base.read_bytes()==control.read_bytes()
models=[]
for run in ['r1','r2']:
    for rank in [0,1]:
        a=read(f'perf2-target-original-v2-{run}-rank{rank}.json')
        b=read(f'perf2-target-fixed-v2-{run}-rank{rank}.json')
        assert a['state_before_sha256']==b['state_before_sha256']
        assert a['first_output_sha256']==b['first_output_sha256']
        assert a['repeat_output_equal'] and b['repeat_output_equal']
        assert a['input_ids']==b['input_ids'] and a['positions']==b['positions'] and a['history']==b['history']
        assert len(set(a['output_hashes']+b['output_hashes']))==1
        models.append(dict(run=run,rank=rank,original_ms=a['median_ms'],fixed_ms=b['median_ms'],
            change_pct=100*(b['median_ms']/a['median_ms']-1),original_range=[a['min_ms'],a['max_ms']],
            fixed_range=[b['min_ms'],b['max_ms']],state_sha256=a['state_before_sha256'],
            output_sha256=a['first_output_sha256'],all_48_output_hashes_equal=True))
micro=read('perf2-gdn-matrix-r0.json')
minimal={}
for label in ['ir2-minimal-multi-r0','ir2-minimal-single-r0','ir2-research-minimal-r0','ir2-research-minimal-single-r0']:
    d=read(label+'.json');rs=d['results']
    minimal[label]=dict(iterations=len(rs),bad=sum(r['state_bad']>0 for r in rs),
        output_bad=sum(not r['output_close'] for r in rs),max_abs=max(r['max_abs'] for r in rs))
pipe=read('host2-cpu18-r0.json')['records']
sleeps=[(r['join_ns']-r['sent_ns'])/1e6 for r in pipe]
summary=dict(base='5f1b1b3c12b1c198813f65275f778c7ad292fb02',fixed_execution='d2d933dad7c6eb55b26ad5b83667c2077e23ca6c',
    compiler_commit='042c923d02a22e5845cf10311cf2d192b17a5d77',control_binary_identical=True,
    control_binary_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),replay_cases=cases,
    error_pattern=read('ir2-exact-state-pattern-r1.json'),minimal=minimal,
    microbench=[{k:c[k] for k in ['batch','queries','correctness','summary']} for c in micro['cases']],
    frozen_backbone=models,
    excluded_frozen_runs=['original-r0: actor class serialization failure',
        'original-r1/r2 and fixed-r0/r1: metadata aliasing in repeated diagnostic call, output hashes differed; not accepted timing evidence',
        'original-v2-r0: local variable shadowed copy module before measurement'],
    host=dict(permission_report='inventory/host2-permission-limits-r0.json',actual_callback_capture_executed=False,
        callback_identity=None,permanent_host_fix=False,preflight='blocked as expected',
        current_cpu18_control=dict(iterations=len(pipe),sleep_over50=sum(t>50 for t in sleeps),max_sleep_ms=max(sleeps))),
    limitations=['Compiler source candidate not built or broad-regression tested',
                 'IR binary repairs only validated on the frozen recorded shape',
                 'Frozen backbone excludes lm_head, draft, rejection, RPC and request scheduling; not serving TPS/TTFT',
                 'No host kernel callback trace; no H200/x86 validation'])
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(replay_cases={k:{x:v[x] for x in ['iterations','state_bad','output_bad','max_abs']} for k,v in cases.items()},minimal=minimal,frozen_backbone=models,host=summary['host']),indent=2))
