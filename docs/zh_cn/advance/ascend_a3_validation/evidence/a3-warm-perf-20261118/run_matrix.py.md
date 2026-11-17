```python
"""Fixed-count AB/BA sessions; refuse incomplete warmup or unmatched token work."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = {
    'agents-historical': dict(kind='http', model='Agents-A1', mtp=0, tokens=128, compiler='legacy',
                              old='baseline', new='native-history'),
    'agents-current': dict(kind='http', model='Agents-A1', mtp=0, tokens=128, compiler='ir2',
                           old='baseline', new='current'),
    'sage-current': dict(kind='http', model='Sage-A1.5', mtp=3, tokens=128, compiler='ir2',
                         old='baseline', new='current'),
    'agents-norm': dict(kind='pipeline', model='Agents-A1', mtp=0, tokens=256, compiler='legacy',
                        old='norm-before', new='norm-after'),
    'sage-norm': dict(kind='pipeline', model='Sage-A1.5', mtp=0, tokens=64, compiler='legacy',
                      old='norm-before', new='norm-after'),
}


def token_map(data, kinds=('measure','qualify')):
    result = {}
    for row in data['token_traces']:
        info = data['phases'][str(row['phase'])]
        if info['kind'] not in kinds:
            continue
        key = (info['kind'], info['batch'], info['rep'], tuple(row['input_ids']))
        assert key not in result
        assert row['finished'] and len(row['token_ids']) == data['options']['tokens']
        result[key] = row['token_ids']
    expected = sum(e['batch'] for e in data['events'] if e['kind'] in kinds)
    assert len(result) == expected, (len(result), expected)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', choices=COMPARISONS, required=True)
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--start-block', type=int, default=0)
    parser.add_argument('--label', required=True)
    a = parser.parse_args()
    cfg = COMPARISONS[a.comparison]
    progress = ROOT / 'results' / f'{a.label}-progress.json'
    assert not progress.exists()
    data = {'comparison': a.comparison, 'config': cfg, 'blocks': [], 'sessions': [], 'complete': False,
            'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'planned_pairs': a.blocks, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():progress.write_text(json.dumps(data, indent=2) + '\n')
    save()
    for block in range(a.start_block, a.start_block+a.blocks):
        pair = {}
        for side in (['old','new'] if block%2==0 else ['new','old']):
            label = f'{a.label}-{side}-b{block:02d}'
            script = ROOT/'scripts'/('http_warm_case.py' if cfg['kind']=='http' else 'pipeline_warm_case.py')
            args = ['--model', '/afs-weights/'+cfg['model'], '--label',label,
                    '--tokens',str(cfg['tokens']), '--block',str(block)]
            if cfg['kind']=='http':args += ['--mtp',str(cfg['mtp'])]
            command = [sys.executable, str(ROOT/'scripts/run_case.py'), '--arm',cfg[side],
                       '--compiler',cfg['compiler'], '--label',label,'--timeout','2400','--',str(script),*args]
            print('START',label,flush=True)
            with (ROOT/'logs'/f'{label}-launcher.log').open('x') as output:
                rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
            data['sessions'].append({'label':label,'side':side,'block':block,'command':command,'returncode':rc})
            save()
            assert rc==0,(label,rc)
            result=json.loads((ROOT/'results'/f'{label}.json').read_text())
            assert result['complete'] and result['warmup_qualified'] and result['measurement_qualified'],label
            assert len(result['worker_summaries'])==2,(label,len(result['worker_summaries']))
            if cfg[side]!='baseline':assert all(not w['dlinfer_modules'] for w in result['worker_summaries'])
            pair[side]=result
            print('DONE',label,flush=True)
        old,new=token_map(pair['old']),token_map(pair['new'])
        differences=[{'kind':k[0],'batch':k[1],'rep':k[2],
                      'input_sha256':hashlib.sha256(json.dumps(k[3]).encode()).hexdigest()}
                     for k in set(old)|set(new) if old.get(k)!=new.get(k)]
        record={'block':block,'order':['old','new'] if block%2==0 else ['new','old'],
                'responses':len(old),'exact_token_work':not differences,'differences':differences}
        data['blocks'].append(record);save()
        assert not differences,record
        print('PAIR_VERIFIED',block,len(old),flush=True)
    data['complete']=True;data['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
```
