"""Plot actual SSE arrivals on 32 active streams during the same cold prefill burst."""

import argparse
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_scheduling import read_json, streams, tpot


def plot(control, local, aligned, repeat, output):
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.9), sharex=True, layout='constrained')
    left = -0.23
    latest = 0
    policies = [(control, 'Original scheduler', '#21537c'), (local, '4K busy prefill, local turns', '#096650'),
                (aligned, '4K busy prefill, DP-aligned turns', '#734190')]
    for axis, (run, label, color) in zip(axes, policies):
        tag = f'probe32-r{repeat}-with-prefill'
        fixed = run / 'fixed'
        probes = streams(fixed / f'{tag}-probes.jsonl.gz', 32, 1024, 512)
        extras = streams(fixed / f'{tag}-prefills.jsonl.gz', 24, 8192, 1)
        marker = read_json(fixed / f'{tag}-summary.json')['injection_marker_s']
        arrivals = [[r['started_s'] + t - marker for t, _ in r['arrivals_s']] for r in probes]
        assert all(events[0] < 0 < events[-1] for events in arrivals)
        latest = max(latest, max(events[-1] for events in arrivals))
        last_prefill = max(r['finished_s'] for r in extras) - marker
        axis.axvspan(0, last_prefill, color='#f4dfbf', alpha=0.68, linewidth=0)
        axis.eventplot(arrivals, colors=color, lineoffsets=range(1, 33), linelengths=0.78, linewidths=0.48)
        axis.axvline(0, color='#a15615', linewidth=1.1)
        gap = median(1000 * max(b - a for a, b in zip(events, events[1:])) for events in arrivals)
        request_tpot = median(tpot(r) for r in probes)
        axis.set_title(f'{label}     |     median max gap {gap:.0f} ms     |     median TPOT {request_tpot:.2f} ms',
                       fontsize=10.5, loc='left', pad=5)
        axis.set(ylabel='Stream', ylim=(0, 33), yticks=(1, 8, 16, 24, 32))
        axis.spines[['top', 'right']].set_visible(False)
    axes[-1].set(xlim=(left, latest + 0.04), xlabel='Seconds from injection of 24 cold 8K prompts')
    fig.suptitle('MTP3 on 8 H200s  ·  Each tick is one nonempty streamed chunk', fontsize=13)
    fig.savefig(output, dpi=150, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('local', type=Path)
    parser.add_argument('aligned', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repeat', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    plot(args.control, args.local, args.aligned, args.repeat, args.output)
