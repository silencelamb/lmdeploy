```python
"""Hold only our outer queue; let the active matrix finish, supplement, then resume.

The inference process, its worker processes, and all production await ordering
remain untouched. Even a failed supplementary probe resumes the original queue.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_matrix import token_map

ROOT=Path(__file__).resolve().parents[1]


def proc(pid):
    path=Path(f'/proc/{pid}')
    if not path.exists():return None
    fields=(path/'stat').read_text().split()
    return {'pid':pid,'ppid':int(fields[3]),'state':fields[2],'starttime':fields[21],
            'command':(path/'cmdline').read_bytes().decode().replace('\0',' ')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True)
    p.add_argument('--matrix',type=int,required=True);a=p.parse_args()
    parent,child=proc(a.parent),proc(a.matrix)
    assert str(ROOT/'scripts/run_remaining.py') in parent['command']
    assert str(ROOT/'scripts/run_matrix_current.py') in child['command'] and '--comparison agents-current' in child['command']
    assert child['ppid']==a.parent
    state={'parent':parent,'child':child,'steps':[],'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    path=ROOT/'results/draft-boundary-r0.json';assert not path.exists()
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    save()
    try:
        os.kill(a.parent,signal.SIGSTOP)
        state['queue_held']=True;save();print('HELD_OUTER_QUEUE',a.parent,flush=True)
        current=proc(a.matrix)
        assert current and current['starttime']==child['starttime'] and current['ppid']==a.parent
        while current and current['starttime']==child['starttime'] and current['state']!='Z':
            time.sleep(5);current=proc(a.matrix)
        state['active_matrix_finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        pair={};coverage={}
        for side,arm in [('old','baseline'),('new','current')]:
            assert proc(a.parent)['state']=='T','Outer queue was resumed externally; do not overlap models'
            label=f'sage-draft-coverage-{side}-r0'
            command=[sys.executable,str(ROOT/'scripts/run_case_current.py'),'--arm',arm,'--compiler','stack91',
                '--label',label,'--timeout','2400','--',str(ROOT/'scripts/http_draft_shape_case.py'),
                '--model','/afs-weights/Sage-A1.5','--mtp','3','--tokens','128','--label',label]
            row={'label':label,'command':command,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            state['steps'].append(row);save();print('START',label,flush=True)
            with (ROOT/'logs'/f'{label}-boundary.log').open('x') as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            row.update(returncode=rc,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
            assert rc==0,(label,rc)
            d=json.loads((ROOT/'results'/f'{label}.json').read_text())
            assert d['complete'] and d['warmup_qualified'] and d['measurement_qualified']
            assert len(d['worker_summaries'])==2
            for worker in d['worker_summaries']:
                assert any(x['signature'][0].startswith('draft:') for x in worker['counts'])
                if side=='new':assert not worker['dlinfer_modules']
            pair[side]=token_map(d)
            original=json.loads((ROOT/'results'/f'sage-current-r0-{side}-b00.json').read_text())
            assert pair[side]==token_map(original),'Supplement output differs from primary fixed workload'
            coverage[side]={str(w['rank']):sorted({str(x['signature']) for x in w['counts']}) for w in d['worker_summaries']}
            print('DONE',label,flush=True)
        assert pair['old']==pair['new']
        state.update(coverage_passed=True,paired_responses=len(pair['old']),shapes=coverage,
                     performance_timings_excluded=True)
        save()
    finally:
        now=proc(a.parent)
        if now and now['starttime']==parent['starttime']:
            os.kill(a.parent,signal.SIGCONT)
            state['queue_resumed']=True
        state['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print('RESUMED_OUTER_QUEUE',a.parent,flush=True)


if __name__=='__main__':main()
```
