# Copyright (c) OpenMMLab. All rights reserved.
"""Pair host trace events; retain unmatched boundaries and missed-probe limits."""

import argparse
import collections
import json
import re
from pathlib import Path

LINE = re.compile(r"^\s*(.+)-(\d+)\s+.*?\[(\d+)\]\s+.*?\s(\d+\.\d+):\s+(\S+):\s*(.*)$")


def analyze(trace, callbacks, metadata):
    pending = collections.defaultdict(list)
    pairs = []
    unmatched = []
    count = collections.Counter()
    irq_stack = collections.defaultdict(list)
    scheduled = collections.defaultdict(list)
    for line in trace.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        task, pid, cpu, ts, event, body = m.groups()
        pid = int(pid)
        cpu = int(cpu)
        t = float(ts)
        event = event.split("/")[-1].split(":")[-1]
        count[event] += 1
        if event in ("tasklet_schedule", "tasklet_hi_schedule"):
            fields = dict(re.findall(r"(tasklet|callback|data|use_cb)=(0x[0-9a-fA-F]+|[0-9]+)", body))
            if len(fields) == 4:
                values = {k: int(v, 0) for k, v in fields.items()}
                arg = values["tasklet"] if values["use_cb"] else values["data"]
                scheduled[(cpu, values["callback"], arg)].append(
                    dict(time_s=t, **fields, origin_irq=irq_stack[cpu][-1] if irq_stack[cpu] else None)
                )
            continue
        if event in ("softirq_entry", "softirq_exit"):
            vec = re.search(r"vec=(\d+)", body)
            if not vec:
                continue
            kind = "softirq"
            identifier = vec[1]
            enter = event.endswith("_entry")
        elif event in ("irq_handler_entry", "irq_handler_exit"):
            irq = re.search(r"irq=(\d+)", body)
            if not irq:
                continue
            kind = "irq"
            identifier = irq[1]
            enter = event.endswith("_entry")
            if enter:
                irq_stack[cpu].append(dict(irq=int(identifier), entry=body, start_s=t))
            elif irq_stack[cpu] and irq_stack[cpu][-1]["irq"] == int(identifier):
                irq_stack[cpu].pop()
        elif re.fullmatch(r"cb\d+_(entry|exit)", event):
            kind = "callback"
            identifier = event.rsplit("_", 1)[0]
            enter = event.endswith("_entry")
        else:
            continue
        key = (cpu, pid, kind, identifier)
        if enter:
            row = dict(start_s=t, cpu=cpu, pid=pid, task=task, kind=kind, id=identifier, entry=body)
            if kind == "callback":
                arg = re.search(r"arg0=(0x[0-9a-fA-F]+)", body)
                address = callbacks.get(identifier, {}).get("address")
                if arg and address:
                    queue = scheduled[(cpu, int(address, 16), int(arg[1], 16))]
                    if queue:
                        row["scheduled_from"] = queue.pop(0)
            pending[key].append(row)
        elif pending[key]:
            row = pending[key].pop()
            row.update(end_s=t, duration_ms=(t - row["start_s"]) * 1000)
            if kind == "callback":
                row["callback"] = callbacks.get(identifier, {"unresolved_event": identifier})
            pairs.append(row)
        else:
            unmatched.append(dict(kind=kind, id=identifier, cpu=cpu, pid=pid, end_s=t, reason="exit_without_entry"))
    for rows in pending.values():
        for row in rows:
            unmatched.append(row | {"reason": "entry_without_exit"})
    callbacks_timed = [r for r in pairs if r["kind"] == "callback"]
    softirq = [r for r in pairs if r["kind"] == "softirq"]
    irqs = [r for r in pairs if r["kind"] == "irq"]
    for callback in callbacks_timed:
        nested = sorted(
            (r["start_s"], r["end_s"])
            for r in irqs
            if r["cpu"] == callback["cpu"] and r["start_s"] >= callback["start_s"] and r["end_s"] <= callback["end_s"]
        )
        merged = []
        for start, end in nested:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        irq_ms = sum(end - start for start, end in merged) * 1000
        callback["nested_irq_ms"] = irq_ms
        callback["excluding_recorded_irq_ms"] = callback["duration_ms"] - irq_ms
    slow = []
    for s in softirq:
        if s["duration_ms"] < 50:
            continue
        enclosed = [
            r
            for r in callbacks_timed
            if r["cpu"] == s["cpu"] and r["start_s"] >= s["start_s"] and r["end_s"] <= s["end_s"]
        ]
        slow.append(s | dict(callbacks=sorted(enclosed, key=lambda r: r["duration_ms"], reverse=True)))
    by_callback = {}
    for r in callbacks_timed:
        by_callback.setdefault(r["id"], []).append(r["duration_ms"])
    stats = {
        key: dict(callback=callbacks.get(key), count=len(times), max_ms=max(times), over50=sum(t > 50 for t in times))
        for key, times in by_callback.items()
    }
    misses = []
    for line in metadata.get("probe_profile", "").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            try:
                if int(fields[-1]) > 0:
                    misses.append(line)
            except ValueError:
                pass
    overruns = {
        key: int(value)
        for key, value in re.findall(
            r"(overrun|commit overrun|dropped events):\s*(\d+)", metadata.get("buffer_stats", "")
        )
    }
    return dict(
        event_counts=dict(count),
        callback_statistics=stats,
        slow_softirq=slow,
        complete_pairs=pairs,
        unmatched=unmatched,
        probe_misses=misses,
        buffer_loss=overruns,
        capture_errors=metadata.get("errors", []),
        cleanup_errors=metadata.get("cleanup_errors", []),
        limitations=[
            "Only callbacks observed during discovery were attached for phase2.",
            "CPU18-only trace omits cross-CPU wakeup sources; unmatched boundary events are excluded from durations.",
            "Callback entry/exit brackets function time, including nested interrupts; "
            "inspect IRQ pairs before blaming callback work.",
            "A correlation without timed callback entry/exit is not callback identification.",
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    d = a.directory
    out = d / "analysis.json"
    assert not out.exists(), out
    result = analyze(
        (d / "trace.txt").read_text(),
        json.loads((d / "callbacks.json").read_text()),
        json.loads((d / "metadata.json").read_text()),
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["callback_statistics", "probe_misses", "buffer_loss", "capture_errors"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
