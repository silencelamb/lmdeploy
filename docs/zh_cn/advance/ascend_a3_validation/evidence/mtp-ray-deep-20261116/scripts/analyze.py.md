# analyze.py

```python
"""Summarize preserved MTP policy audits and same-host HTTP/RPC timelines."""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def save(name, data):
    (ROOT/'results'/name).write_text(json.dumps(data, indent=2)+'\n')


def mtp():
    summary = dict(audits=[])
    for p in sorted((ROOT/'results').glob('mtp-runtime-*.jsonl')):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        summary['audits'].append(dict(file=p.name, calls=len(rows),
            request_rows=sum(x['shape'][0] for x in rows),
            greedy_all=all(x['max_top_k']==1 for x in rows),
            cpu_rng_all_unchanged=all(x['cpu_rng_unchanged'] for x in rows),
            npu_rng_all_unchanged=all(x['npu_rng_unchanged'] for x in rows),
            oracle_all_equal=all(x['cpu_oracle_equal'] for x in rows),
            no_dlinfer=all(not x['dlinfer_imported'] for x in rows),
            rejected_count_histogram=dict(collections.Counter(
                n for x in rows for n in x['num_rejected_tokens']))))
    a, b = [load(ROOT/f'results/mtp3-greedy-runtime-r{i}.json') for i in range(2)]
    differences = []
    equal = total = 0
    for e, f in zip(a['events'], b['events'], strict=True):
        assert e['label'] == f['label'] and e['prompts'] == f['prompts']
        for i, (x, y) in enumerate(zip(e['responses'], f['responses'], strict=True)):
            total += 1
            if x['token_ids'] == y['token_ids']:
                equal += 1
                continue
            pos = next(j for j, (u, v) in enumerate(zip(x['token_ids'], y['token_ids'])) if u != v)
            differences.append(dict(event=e['label'], row=i, first_diff=pos,
                ids=[x['token_ids'][pos], y['token_ids'][pos]],
                logprobs=[x['logprobs'][pos], y['logprobs'][pos]]))
    summary.update(complete=[a['complete'], b['complete']], responses=total,
                   exact_equal=equal, differences=differences)
    save('mtp-runtime-summary.json', summary)
    for label, data in [('r0', a), ('r1', b)]:
        rows = [json.loads(line) for line in
                (ROOT/f'results/mtp-runtime-mtp3-greedy-{label}-rank0.jsonl').read_text().splitlines()]
        response = data['events'][0]['responses'][0]['token_ids']
        output = response[:1]  # First token is selected by the prefill sampler.
        observations = []
        for row in rows:
            assert row['shape'][0] == 1
            for pos, token in enumerate(row['output_token_ids'][0]):
                if token < 0:
                    break
                observations.append(dict(token_index=len(output), token_id=token,
                    sequence=row['sequence'], position_in_target_block=pos,
                    top2_ids=row['target_top2_ids'][0][pos],
                    top2_values=row['target_top2_values'][0][pos], drafts=row['draft_token_ids'][0]))
                output.append(token)
            if len(output) >= len(response):
                break
        assert output[:len(response)] == response
        save(f'first-single-{label}-target-trace.json', observations)
    print('MTP', total, 'responses;', equal, 'exact;',
          sum(x['request_rows'] for x in summary['audits']), 'request-level verifier checks')


def rpc(label, http_name):
    records = []
    for p in (ROOT/'results').glob(f'ray-{label}-*.jsonl'):
        records.extend(json.loads(line) for line in p.read_text().splitlines())
    http = load(ROOT/'results'/http_name)
    beats = [r for r in records if r['kind'] == 'heartbeat_late']
    deserialized = {(r['driver_pid'], r['item'], r['rank']): r for r in records
                    if r['kind'] == 'driver_ack_deserialized'}
    waits = []
    puts = {(r['driver_pid'], r['item']): r for r in records if r['kind']=='driver_put_exit'}
    forwards = {(r['pid'], r['item']): r for r in records if r['kind']=='driver_forward_exit'}
    for r in records:
        if r['kind'] != 'driver_gather_done':
            continue
        batch = next((b for b in http['events'] if b['start_ns'] <= r['done_ns'] <= b['end_ns']), None)
        row = dict(item=r['acknowledgements'][0]['item'], driver_pid=r['pid'],
                   began_ns=r['began_ns'], done_ns=r['done_ns'], wait_ms=r['wait_ms'],
                   thread_cpu_ms=r['thread_cpu_ms'], workers=[])
        put = puts.get((row['driver_pid'], row['item']))
        forward = forwards.get((row['driver_pid'], row['item']))
        if put:
            row['ray_put_ms'] = (put['t_ns']-put['dispatch_ns'])/1e6
        if forward:
            row['submission_function_ms'] = forward['duration_ms']
        if batch:
            row.update(concurrency=batch['concurrency'], repeat=batch['repeat'], warmup=batch['warmup'])
        for ack in r['acknowledgements']:
            d = deserialized.get((ack['driver_pid'], ack['item'], ack['rank']))
            worker = dict(rank=ack['rank'], worker_pid=ack['worker_pid'],
                          dispatch_ns=ack['dispatch_ns'], enter_ns=ack['worker_enter_ns'],
                          exit_ns=ack['worker_exit_ns'],
                          dispatch_to_enter_ms=(ack['worker_enter_ns']-ack['dispatch_ns'])/1e6,
                          handler_ms=(ack['worker_exit_ns']-ack['worker_enter_ns'])/1e6,
                          exit_to_gather_ms=(r['done_ns']-ack['worker_exit_ns'])/1e6)
            if d:
                worker.update(deserialize_enter_ns=d['deserialize_enter_ns'],
                    deserialize_exit_ns=d['deserialize_exit_ns'],
                    exit_to_deserialize_ms=(d['deserialize_enter_ns']-ack['worker_exit_ns'])/1e6,
                    deserialize_ms=(d['deserialize_exit_ns']-d['deserialize_enter_ns'])/1e6,
                    deserialize_to_gather_ms=(r['done_ns']-d['deserialize_exit_ns'])/1e6)
            row['workers'].append(worker)
        row['overlapping_heartbeat_delays'] = [beat for beat in beats
            if beat['actual_ns'] >= r['began_ns'] and beat['expected_ns'] <= r['done_ns']]
        waits.append(row)
    gaps = []
    for batch in http['events']:
        for index, response in enumerate(batch['responses']):
            for i, (before, after) in enumerate(zip(response['events'], response['events'][1:]), start=1):
                gap = (after['t_ns']-before['t_ns'])/1e6
                if gap <= 100:
                    continue
                overlapping = [w for w in waits if w['wait_ms'] > 50 and
                               w['done_ns'] >= before['t_ns'] and w['began_ns'] <= after['t_ns']]
                gaps.append(dict(concurrency=batch['concurrency'], repeat=batch['repeat'],
                    warmup=batch['warmup'], response=index, chunk_index=i, gap_ms=gap,
                    began_ns=before['t_ns'], done_ns=after['t_ns'],
                    slow_rpc_items=[w['item'] for w in overlapping]))
    long_waits = sorted((w for w in waits if w['wait_ms']>50), key=lambda x:x['wait_ms'], reverse=True)
    summary = dict(http_complete=http['complete'], http_batches=len(http['events']),
        measured_requests=sum(len(b['responses']) for b in http['events'] if not b['warmup']),
        rpc_count=len(waits), record_count=len(records),
        setup=[r for r in records if r['kind']=='driver_setup'],
        long_rpc_waits=long_waits, sse_gaps_over_100ms=gaps,
        heartbeat_delays=beats,
        limitations=['Probe changes acknowledgement payload and adds timestamps/heartbeats.',
                     'SSE chunks are not necessarily one token; first gaps may include prefill/capture.',
                     'worker exit -> deserialization does not by itself isolate Ray C++ or transport.'])
    save(f'{label}-summary.json', summary)
    print('HTTP', summary['http_complete'], summary['measured_requests'], 'measured requests;',
          len(waits), 'RPC joins;', len(long_waits), 'waits >50ms')
    for row in long_waits[:5]:
        print(json.dumps(row))
    print('steady measured SSE gaps', len([g for g in gaps if not g['warmup'] and g['chunk_index']>8]))


def cpu_control():
    path = ROOT/'results/ray-cpu-control-r0.json'
    if not path.exists():
        return
    data = load(path)
    summary = {k:data[k] for k in ('ray_version', 'cpu_arch', 'torch_imported', 'npu_imported', 'complete')}
    summary['modes'] = {}
    for mode in ('put', 'direct'):
        rows = [r for r in data['records'] if r['mode']==mode]
        delivery = [(q['entered_ns']-q['dispatch_ns'])/1e6 for r in rows for q in r['replies']]
        summary['modes'][mode] = dict(joins=len(rows), max_wait_ms=max(r['wait_ms'] for r in rows),
            max_delivery_ms=max(delivery), delivery_over100ms=sum(t>100 for t in delivery),
            all_workers_without_torch=all(not q['torch_imported'] for r in rows for q in r['replies']),
            all_workers_without_npu=all(not q['npu_imported'] for r in rows for q in r['replies']))
    summary['slow_events'] = []
    for row in data['records']:
        delivery = [(q['entered_ns']-q['dispatch_ns'])/1e6 for q in row['replies']]
        if row['wait_ms']>100 or max(delivery)>100:
            summary['slow_events'].append(dict(item=row['replies'][0]['item'], mode=row['mode'],
                block=row['block'], wait_ms=row['wait_ms'], delivery_ms=delivery,
                driver_wait_after_dispatch_ms=(row['began_ns']-row['replies'][0]['dispatch_ns'])/1e6,
                entry_to_done_ms=[(row['done_ns']-q['entered_ns'])/1e6 for q in row['replies']]))
    save('ray-cpu-control-summary.json', summary)
    print('CPU-only', len(data['records']), 'joins;', len(summary['slow_events']), 'slow events')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ray-label')
    p.add_argument('--http-file')
    args = p.parse_args()
    mtp()
    cpu_control()
    if args.ray_label:
        rpc(args.ray_label, args.http_file)
```
