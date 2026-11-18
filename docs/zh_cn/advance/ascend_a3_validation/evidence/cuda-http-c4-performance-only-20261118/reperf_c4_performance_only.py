"""Separate HTTP timing evidence from the unchanged bitwise-correctness gate."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics as st

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--campaign', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
root = a.root
campaign = a.campaign or root / 'h200-reperf-20261117/clean-20261118'
out = a.output or root / 'h200-reperf-20261117/http-c4-performance-only-20261118'
out.mkdir(parents=True, exist_ok=True)
raw = json.loads((campaign / 'performance-raw.json').read_text())
original = json.loads((campaign / 'performance-summary.json').read_text())
resources = {r['result']: r for r in json.loads((campaign / 'resource-audit.json').read_text())['runs']}
all_rows = []
input_files = set()
for model in ('flash', 's2', 'glm'):
    records = [r for r in raw if r['model'] == model and r['scope'] == 'http'
               and r['concurrency'] == 4 and r['mtp'] == 0]
    assert len(records) == 10
    assert {(r['pair'], r['role']) for r in records} == {(i, v) for i in range(5) for v in ('base', 'pr')}
    expected = next(r for r in original if r['model'] == model and r['scope'] == 'http'
                    and r['concurrency'] == 4 and r['mtp'] == 0)
    config = records[0]['inference_config']
    assert config['gpu_ids'] == '0,1,2,3,4,5,6,7'
    cmd = config['server_command']
    assert cmd[cmd.index('--tp') + 1] == '8' and cmd[cmd.index('--mtp') + 1] == '0'
    assert '--prefix' not in cmd and '--eager' not in cmd
    prompt_reference = None
    batch_rows, run_rows, first_differences = [], [], {}
    refs = {'base': {}, 'pr': {}}
    peaks = {'base': [], 'pr': []}
    for r in sorted(records, key=lambda r: (r['pair'], r['role'])):
        assert r['inference_config'] == config
        audit = resources[r['result']]
        assert audit['no_sampled_overlap'] and audit['all_import_audits_ok'] and audit['process_completed_as_expected']
        path = root / 'results' / r['result']
        input_files.add(path)
        data = json.loads(path.read_text())
        assert data['complete'] and data['mtp'] == 0
        events = [e for e in data['events'] if e.get('batch') == 4 and e.get('warmup') is False]
        warmups = [e for e in data['events'] if e.get('batch') == 4 and e.get('warmup') is True]
        assert len(events) == 5 and len(warmups) >= 3
        measured = []
        for e in events:
            assert len(e['responses']) == len(e['prompts']) == 4
            # Match input token IDs and positions, independently of generated IDs.
            inputs = [(prompt, response['prompt_token_ids']) for prompt, response in zip(e['prompts'], e['responses'])]
            if prompt_reference is None:
                prompt_reference = inputs
            assert inputs == prompt_reference
            for index, response in enumerate(e['responses']):
                assert response['cached_tokens'] == 0
                assert response['input_token_len'] == len(response['prompt_token_ids'])
                assert len(response['token_ids']) == response['usage']['completion_tokens'] == 256
                assert response['finish_reason'] == 'length'
                reference = refs[r['role']].setdefault(index, (r['result'], e['label'], response['token_ids']))
                if response['token_ids'] != reference[2] and r['role'] not in first_differences:
                    token_index = next(i for i, (x, y) in enumerate(zip(reference[2], response['token_ids'])) if x != y)
                    first_differences[r['role']] = {
                        'reference_result': reference[0], 'reference_event': reference[1],
                        'result': r['result'], 'event': e['label'], 'request_index': index,
                        'first_token_index_zero_based': token_index,
                        'reference_token': reference[2][token_index], 'repeat_token': response['token_ids'][token_index],
                        'identical_previous_tokens': reference[2][:token_index]}
            value = {'throughput': e['throughput'], 'ttft': st.mean(x['ttft'] for x in e['responses']),
                     'tpot': st.mean(x['tpot'] for x in e['responses'])}
            assert value['throughput'] == 1024 / e['elapsed']
            original_batch = next(x for x in r['measurements'] if x['event'] == e['label'])
            assert value == original_batch['metrics']
            controls = [x for x in data['capture_controls'] if x['time_ns'] < e['start_unix_ns']]
            assert controls[-1]['original_generate_restored'] and not controls[-1]['enabled']
            assert not any(e['start_unix_ns'] <= x['time_ns'] <= e['end_unix_ns'] for x in data['capture_controls'])
            peaks[r['role']].extend(original_batch['peak_gpu_mib'].values())
            measured.append(value)
            batch_rows.append({'model': model, 'pair': r['pair'], 'role': r['role'], 'result': r['result'],
                               'event': e['label'], 'input_tokens': sum(len(x[1]) for x in inputs),
                               'output_tokens': 1024, 'cache_hits': 0, **value})
        metric = {k: st.median(x[k] for x in measured) for k in ('throughput', 'ttft', 'tpot')}
        assert metric == r['metrics']
        run_rows.append({'pair': r['pair'], 'role': r['role'], 'metrics': metric})
    assert len(batch_rows) == 50 and set(first_differences) == {'base', 'pr'}
    metrics = {}
    for metric in ('throughput', 'ttft', 'tpot'):
        left = [next(x for x in run_rows if x['pair'] == i and x['role'] == 'base')['metrics'][metric] for i in range(5)]
        right = [next(x for x in run_rows if x['pair'] == i and x['role'] == 'pr')['metrics'][metric] for i in range(5)]
        delta = [(b / a - 1) * 100 for a, b in zip(left, right)]
        rng = random.Random(42)
        boot = sorted(st.median(rng.choices(delta, k=5)) for _ in range(10000))
        ci = [boot[250], boot[9749]]
        sign = -1 if metric == 'throughput' else 1
        adverse = sum(x * sign > 5 for x in delta)
        trigger = st.median(delta) * sign > 5 and adverse >= 4 and min(x * sign for x in ci) > 0
        prior = expected['metrics'][metric]
        assert left == prior['base'] and right == prior['pr']
        assert delta == prior['paired_pct'] and ci == prior['ci95'] and trigger == prior['requires_investigation']
        metrics[metric] = {**prior, 'adverse_pairs_above_5pct': adverse}
    all_rows.append({'model': model, 'scope': 'real HTTP loopback C4; performance-only workload comparison',
        'pair_count': 5, 'process_count': 10, 'timed_batches': 50, 'responses': 200, 'generated_tokens': 51200,
        'input_tokens_per_request': [len(x[1]) for x in prompt_reference], 'output_tokens_per_request': 256,
        'mtp': 0, 'graph': True, 'prefix': False, 'tp': 8, 'inference_config': config,
        'same_prompt_ids_order_lengths_cache_and_config': True, 'all_measured_resources_clean': True,
        'capture_disabled_during_timing': True, 'first_replay_token_differences': first_differences,
        'bitwise_correctness_qualified': False,
        'performance_only_criterion_passed': not any(m['requires_investigation'] for m in metrics.values()),
        'sampled_peak_mib_max_single_gpu': {role: max(v) if v else None for role, v in peaks.items()},
        'metrics': metrics, 'timed_batches_raw': batch_rows})
sources = [campaign / name for name in ('performance-raw.json', 'performance-summary.json', 'resource-audit.json')]
manifest = [{'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes': f.stat().st_size}
            for f in sorted(input_files | set(sources))]
result = {'report_date': '2026-11-18 Asia/Shanghai',
    'purpose': 'User requested independent performance reporting despite known numerical replay differences.',
    'new_gpu_runs': 0, 'reuses_completed_warmed_abba_measurements': True,
    'numerical_threshold_changed': False, 'timing_threshold_changed': False,
    'not_claimed': ['bitwise HTTP replay equivalence', 'identical internal MoE routes or batching', 'all metrics within ±5%'],
    'processes': 30, 'timed_batches': 150, 'responses': 600, 'generated_tokens': 153600,
    'six_model_dynamic_c4_precision': [
        {k: row[k] for k in ('model', 'paired_processes', 'timed_within_source',
                             'timed_outputs_repeat_and_match', 'score_probes_repeat_and_match',
                             'numerically_qualified')}
        for row in original if row['scope'] == 'http' and row['concurrency'] == 4],
    'rows': all_rows, 'source_files': manifest}
(out / 'performance-only-audit.json').write_text(json.dumps(result, indent=2) + '\n')
for row in all_rows:
    print(row['model'], 'workload audit passed', 'performance', row['performance_only_criterion_passed'],
          {k: round(v['paired_median_pct'], 2) for k, v in row['metrics'].items()},
          'first token differences', {k: v['first_token_index_zero_based'] for k, v in row['first_replay_token_differences'].items()})
