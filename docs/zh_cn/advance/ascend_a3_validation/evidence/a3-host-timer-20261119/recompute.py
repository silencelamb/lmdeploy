"""Rebuild the public report tables from the CSV files beside this script (Python 3 standard library)."""
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def read_rows(name):
    path = ROOT / name
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b, tolerance=1e-6):
    assert math.isclose(a, b, rel_tol=0, abs_tol=tolerance), (a, b)


def main():
    data = json.loads((ROOT / 'measurements.json').read_text())
    scans = read_rows('all-core-scan.csv')
    quiet = read_rows('quiet-wakeups.csv.gz')
    vectors = read_rows('vector-windows.csv')
    context = read_rows('context-windows.csv')
    intervals = read_rows('timing-holdout.csv')
    for node, source in [('A','local'), ('B','remote')]:
        measured = data['nodes'][node]
        cores = [r for r in scans if r['host'] == source]
        assert len({r['cpu'] for r in cores}) == measured['active_cpu_coverage'] == 319
        expected_cpu = 18 if node == 'A' else 44
        assert [int(r['cpu']) for r in cores if int(r['over100'])] == [expected_cpu]
        for cpu, stats in measured['quiet_75s'].items():
            rows = [r for r in quiet if r['node'] == node and r['cpu'] == cpu]
            due = [int(r['deadline_offset_ns']) for r in rows]
            delta = [(int(r['woke_offset_ns']) - t) / 1e6 for r,t in zip(rows,due)]
            long_at = [t for t,d in zip(due,delta) if d > 100]
            assert len(rows) == stats['samples'] and len(long_at) == stats['over100']
            assert sum(d > 300 for d in delta) == stats['over300']
            close(max(delta), stats['max_late_ms'])
            if long_at:
                close(median((b-a)/1e9 for a,b in zip(long_at,long_at[1:])), stats['period_s'])
            print(f'{node} CPU{cpu}: {len(long_at)} / {len(rows)} late by >100 ms, max {max(delta):.1f} ms')
        for phase, stats in measured['fine_100s'].items():
            subset = [r for r in vectors if r['node'] == node and r['stage'] == phase]
            assert len(subset) == stats['by_core'][str(stats['hot_or_test_cpu'])]['over100']
            tight = [r for r in subset if r['tight'] == '1']
            for r in tight:
                assert abs(float(r['entry_skew_ms'])) <= 10 and abs(float(r['exit_skew_ms'])) <= 10
            assert len(tight) == stats['tight']
            assert dict(Counter(r['last_vectors'].replace('+',',') for r in tight)) == stats['tight_last']
        for phase in ('single','dual_intersection'):
            rows = [r for r in context if r['node'] == node and r['phase'] == phase]
            stats = measured['context'][phase]['summary']
            normals = [r for r in rows if r['normal_arch_timer_irq']]
            assert len(normals) == stats['normal_available'] and len(rows) == stats['qualified']
            assert all(int(r['arch_timer_irq']) > 0 for r in rows)
            assert all(int(r['sched_slices']) == int(r['hot_switch_ins']) == int(r['hot_switch_outs']) == 0 for r in rows)
            for r in rows:
                close(float(r['interior_ms']), float(r['own_current_hot_ms']))
            close(median(float(r['witness_gap_ms']) for r in rows), stats['gap_ms'][1])
            close(median(int(r['arch_timer_irq']) for r in rows), stats['inner_irq'][1])
            close(median(int(r['normal_sched_slices']) for r in normals), stats['pre_scheduler_slices'][1])
            print(f'{node} {phase}: {len(rows)} aligned stalls, {len(normals)} normal controls')
        hold = [r for r in intervals if r['node'] == node]
        hits = 0
        for r in hold:
            observed = float(r['observed_start_to_start_ms'])
            err = min(abs(float(r[k])-observed) for k in
                      ('predicted_32ms_from_previous_start_ms','predicted_256ms_from_previous_start_ms'))
            close(err, float(r['nearest_error_ms']))
            assert int(err <= 5) == int(r['within_5ms'])
            hits += err <= 5
        assert hits == measured['holdout']['holdout']['hits5'] and len(hold) == measured['holdout']['holdout']['n']
        print(f'{node} independent timing predictions: {hits}/{len(hold)}')
    assert len(quiet) == 47127 and len(context) == 147 and len(intervals) == 72
    tail = read_rows('original-slow-requests.csv')
    fresh = read_rows('fresh-requests.csv.gz')
    paired = read_rows('paired-placement.csv')
    pair_values = defaultdict(dict)
    for r in paired:
        pair_values[r['pair']][r['side']] = float(r['early_hit_pct'])
    assert len(pair_values) == 8 and all(len(r) == 2 for r in pair_values.values())
    delta = mean(r['new']-r['old'] for r in pair_values.values())
    close(delta, data['software_on_node_A']['paired_position_first_400ms']['delta'])
    for side in ('old','new'):
        selected = [r for r in tail if r['software'] == side]
        covered = [r for r in selected if r['inner_softirq_ms']]
        q = data['software_on_node_A']['original_tail'][side]['tail']
        assert len(selected) == q['n'] and len(covered) == q['covered']
        assert all(float(r['inner_softirq_ms']) >= 300 for r in covered)
        close(median(float(r['inner_softirq_ms']) for r in covered), q['median_inner_ms'])
        selected = [r for r in fresh if r['software'] == side]
        assert len(selected) == 1000
        totals = Counter()
        for r in selected:
            slow = float(r['ttft_ms']) > 1000
            hit = max(float(r[k+'_overlap_ms']) for k in ('core','rpc','client','background')) >= 300
            assert int(slow) == int(r['over_1s']) and int(hit) == int(r['any_role_at_least_300ms'])
            totals[slow,hit] += 1
        q = data['software_on_node_A']['new_series'][side]
        assert totals[True,True] + totals[True,False] == q['over_1s']
        assert totals[True,True] == q['over_1s_with_job_role_runnable_cpu18_and_timer_300ms']
        assert totals[False,True] == q['under_1s_with_same_observation']
        print(f'{side} fresh TTFT >1s: {q["over_1s"]}/1000, core CPU18 early hits: {sum(r[side] for r in pair_values.values()):.0f}/800')


if __name__ == '__main__':
    main()
