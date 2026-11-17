```python
"""Post-run workload/warmup audit, independent of favorable timing results."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    progress = json.loads((ROOT/'results'/f'{a.label}-progress.json').read_text())
    reference = {}
    measured_reference = {}
    out = {'comparison': progress['comparison'], 'complete': progress['complete'],
           'sessions': [], 'qualifier_mismatches': [], 'measurement_replay_mismatches': [], 'errors': []}
    expected_affinity = None
    core_script_hashes = collections.defaultdict(set)
    for session in progress['sessions']:
        label, side = session['label'], session['side']
        path = ROOT/'results'/f'{label}.json'
        data = json.loads(path.read_text())
        meta = json.loads((ROOT/'logs'/f'{label}.json').read_text())
        active_scripts=['warm_observer.py'] + (['warm_serve.py','http_warm_case.py']
            if progress['config']['kind']=='http' else ['pipeline_warm_case.py'])
        for script in active_scripts:
            core_script_hashes[script].add(meta['scripts'][script])
        guard_path=ROOT/'artifacts'/f'{label}-scripts/guard/sitecustomize.py'
        core_script_hashes['guard/sitecustomize.py'].add(hashlib.sha256(guard_path.read_bytes()).hexdigest())
        if expected_affinity is None:
            expected_affinity = meta['cpu_affinity']
        row = {'label': label, 'side': side, 'source_head': meta['source_head'],
               'source_diff_sha256': meta['source_diff_sha256'], 'profile': meta['compiler_profile'],
               'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'warmup_sweeps': len([e for e in data['events'] if e['kind']=='warmup'])//4,
               'input_lengths': collections.Counter(), 'cache_tokens': collections.Counter(),
               'qualified_responses': 0, 'measured_responses': 0, 'forward_counts': {},
               'output_slots': collections.Counter(), 'token_delivery_gaps_over_100ms': [],
               'errors': []}
        if not all(data.get(k) for k in ['complete','warmup_qualified','measurement_qualified']):
            row['errors'].append('incomplete or unqualified measurement')
        if meta['source_diff_sha256'] != hashlib.sha256(b'').hexdigest():
            row['errors'].append('source has local modifications')
        if meta['cpu_affinity'] != expected_affinity or 18 not in meta['cpu_affinity']:
            row['errors'].append('driver CPU affinity differs or excludes CPU18')
        for worker in data.get('worker_summaries', []):
            if worker['affinity'] != expected_affinity:
                row['errors'].append('worker CPU affinity differs')
            if meta['role']=='native' and worker['dlinfer_modules']:
                row['errors'].append('native imported dlinfer')
            counts = collections.Counter()
            for item in worker['counts']:
                info = data['phases'].get(str(item['phase']))
                if info and info['kind']=='measure':
                    counts[str(item['signature'])] += item['count']
            row['forward_counts'][str(worker['rank'])] = dict(counts)
        if len(data.get('worker_summaries', [])) != 2:
            row['errors'].append('missing worker summaries')
        for trace in data.get('token_traces', []):
            info = data['phases'][str(trace['phase'])]
            if info['kind'] not in ['measure','qualify']:
                continue
            row['input_lengths'][len(trace['input_ids'])] += 1
            row['cache_tokens'][trace['cached_tokens']] += 1
            if len(trace['token_ids']) != data['options']['tokens'] or not trace['finished']:
                row['errors'].append('unfinished or wrong output work')
            if trace['cached_tokens'] != 0:
                row['errors'].append('primary workload unexpectedly hit prefix cache')
            if progress['config']['kind']=='http' and len(trace['input_ids']) != 171:
                row['errors'].append('HTTP input length differs from historical171')
            if info['kind']=='qualify':
                row['qualified_responses'] += 1
                key = (side,info['batch'],tuple(trace['input_ids']))
                if key in reference and reference[key]['ids'] != trace['token_ids']:
                    out['qualifier_mismatches'].append({'first':reference[key]['label'],
                        'later':label,'batch':info['batch'],
                        'input_sha256':hashlib.sha256(json.dumps(trace['input_ids']).encode()).hexdigest()})
                reference.setdefault(key, {'label':label, 'ids':trace['token_ids']})
            else:
                row['measured_responses'] += 1
                key=(side,info['batch'],info['rep'],tuple(trace['input_ids']))
                if key in measured_reference and measured_reference[key]['ids'] != trace['token_ids']:
                    out['measurement_replay_mismatches'].append({'first':measured_reference[key]['label'],
                        'later':label,'batch':info['batch'],'rep':info['rep'],
                        'input_sha256':hashlib.sha256(json.dumps(trace['input_ids']).encode()).hexdigest()})
                measured_reference.setdefault(key, {'label':label,'ids':trace['token_ids']})
                for before,after in zip(trace['chunks'],trace['chunks'][1:]):
                    gap = (after[0]-before[0])/1e6
                    if gap > 100:
                        row['token_delivery_gaps_over_100ms'].append({
                            'phase':trace['phase'],'session_id':trace['session_id'],
                            'start_monotonic_ns':before[0],'end_monotonic_ns':after[0],'gap_ms':gap})
        for driver in data.get('driver_summaries', []):
            for item in driver['output_step_histogram']:
                info = data['phases'].get(str(item['phase']))
                if info and info['kind']=='measure':
                    row['output_slots'][str(item['valid_slots_per_row'])] += item['steps']
        for key in ['input_lengths','cache_tokens','output_slots']:
            row[key] = dict(row[key])
        if row['qualified_responses'] != 5 or row['measured_responses'] != 57:
            row['errors'].append('wrong formal response count')
        out['sessions'].append(row)
        out['errors'].extend({'label':label,'error':err} for err in row['errors'])
    out['core_script_sha256']={k:sorted(v) for k,v in core_script_hashes.items()}
    for script,hashes in core_script_hashes.items():
        if len(hashes)!=1:
            out['errors'].append({'error':'core harness changed between sessions','script':script})
    if out['complete'] and len(reference)!=10:
        out['errors'].append({'error':'qualifier inputs differed across processes','unique_keys':len(reference)})
    if out['complete'] and len(measured_reference)!=114:
        out['errors'].append({'error':'measurement inputs differed across processes','unique_keys':len(measured_reference)})
    out['qualified'] = (out['complete'] and not out['errors'] and not out['qualifier_mismatches']
                        and not out['measurement_replay_mismatches']
                        and all(b['exact_token_work'] for b in progress['blocks']))
    output = Path(a.output)
    assert not output.exists()
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'qualified':out['qualified'],'errors':out['errors'],
                      'qualifier_mismatches':out['qualifier_mismatches'],
                      'measurement_replay_mismatches':out['measurement_replay_mismatches'],
                      'sessions':len(out['sessions'])}))


if __name__=='__main__':
    main()
```
