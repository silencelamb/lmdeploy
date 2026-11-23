"""Compare valid AgentX windows and match completed requests by source trajectory and turn."""

import argparse
import json
from pathlib import Path
from statistics import median

from agentx_slo import score


def read_profile(path):
    report = json.loads((path / 'profile_export_aiperf.json').read_text())
    config, meta = report['input_config'], report['metadata']
    dataset = config['datasets'][0]
    window = next(w for w in meta['metric_duration_coverage'] if w['phase_name'] == 'profiling')
    conditions = (meta['scenario'], meta['dataset']['loader'], dataset['random_seed'], dataset['max_context_length'],
                  config['phases'][0]['concurrency'], window['expected_duration_seconds'])
    result = score(path / 'profile_export_aiperf.json', duration_s=conditions[-1])
    turns = {}
    with (path / 'profile_export.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            m, metrics = record['metadata'], record.get('metrics', {})
            if m['benchmark_phase'] != 'profiling' or m.get('was_cancelled') or record.get('error'):
                continue
            if not {'inter_token_latency', 'time_to_first_token', 'output_token_throughput_per_user'} <= metrics.keys():
                continue
            key = tuple(m.get(k) for k in ('conversation_id', 'turn_index', 'source_outer_idx', 'source_inner_idx'))
            if key in turns or key[0] is None or key[1] is None:
                raise ValueError('Cannot uniquely match the scored trajectory turns.')
            values = {k: v['value'] for k, v in metrics.items()}
            turns[key] = dict(ttft=values['time_to_first_token'], tpot=values['inter_token_latency'],
                              shape=(values['usage_prompt_tokens'], values['usage_completion_tokens']),
                              cached=values['usage_prompt_cache_read_tokens'],
                              passes=values['time_to_first_token'] <= 3000
                              and values['output_token_throughput_per_user'] >= 40)
    return conditions, result, turns


def paired(rows):
    if not rows:
        return dict(n=0)
    return dict(n=len(rows), paired_ttft_delta_median_ms=round(median(b['ttft'] - a['ttft'] for a, b in rows), 3),
                paired_tpot_delta_median_ms=round(median(b['tpot'] - a['tpot'] for a, b in rows), 3),
                both_limits_control=sum(a['passes'] for a, _ in rows),
                both_limits_candidate=sum(b['passes'] for _, b in rows))


def compare(control, candidate):
    setup, original, a = read_profile(control)
    new_setup, changed, b = read_profile(candidate)
    if setup != new_setup:
        raise ValueError(f'AgentX replay conditions differ: {setup} vs {new_setup}')
    pairs = [(a[k], b[k]) for k in a.keys() & b.keys()]
    same_shape = [(x, y) for x, y in pairs if x['shape'] == y['shape']]
    same_cache = [(x, y) for x, y in same_shape if x['cached'] == y['cached']]
    return dict(control=original, candidate=changed, matched=dict(
        control_scored_turns=len(a), candidate_scored_turns=len(b), same_turn=paired(pairs),
        same_prompt_and_output_lengths=paired(same_shape), also_same_cached_length=paired(same_cache)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path, help='Directory containing the original AIPerf export.')
    parser.add_argument('candidate', type=Path, help='Directory containing the candidate AIPerf export.')
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate), indent=2, allow_nan=False))
