"""Measure client-visible decode cadence while all 64 first-wave requests are generating."""

import argparse
import gzip
import json
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from statistics import mean, median


def summarize(path):
    case = path / "fixed/i8192-o256-c64-r0"
    config = json.loads((case / "summary.json").read_text())
    assert (path / "exit_code").read_text().strip() == "0"
    assert (config["context"], config["output_tokens"], config["concurrency"], config["success"]) == (
        8192,
        256,
        64,
        256,
    )
    raw = case / "requests.jsonl.gz"
    with gzip.open(raw, "rt") as f:
        rows = sorted((json.loads(line) for line in f), key=lambda row: row["started_s"])
    first = rows[:64]
    next_submit = rows[64]["started_s"]
    workers = Counter(row["url"] for row in first)
    assert len(rows) == 256 and sorted(workers.values()) == [8] * 8
    assert all(row["error"] is None and row["input_tokens"] == 8192 and row["output_tokens"] == 256 for row in rows)
    arrivals = []
    for row in first:
        events = row["arrivals_s"]
        assert len(events) == 256 and all(count == 1 for _, count in events)
        timestamps = [row["started_s"] + delay for delay, _ in events]
        assert timestamps == sorted(timestamps)
        arrivals.append(timestamps)

    last_first_token = max(t[0] for t in arrivals)
    first_last_token = min(t[-1] for t in arrivals)
    windows = {}
    for low, high in ((33, 224), (65, 192)):
        earliest_start = min(t[low - 1] for t in arrivals)
        latest_end = max(t[high - 1] for t in arrivals)
        assert last_first_token < earliest_start < latest_end < first_last_token <= next_submit
        per_request = [1000 * (t[high - 1] - t[low - 1]) / (high - low) for t in arrivals]
        windows[f"token_{low}_to_{high}"] = dict(
            intervals=64 * (high - low),
            all_requests_generating_before_start_s=earliest_start - last_first_token,
            until_first_last_token_after_window_s=first_last_token - latest_end,
            mean_ms_per_token=mean(per_request),
            median_request_ms_per_token=median(per_request),
        )

    first_to_second = [t[1] - t[0] for t in arrivals]
    return dict(
        source=str(raw),
        seed=config["seed"],
        live_requests_per_worker=sorted(workers.values()),
        last_first_token_s=last_first_token,
        first_last_token_s=first_last_token,
        next_submit_s=next_submit,
        first_wave_first_to_second_token_s_p50=median(first_to_second),
        first_wave_tokens_received_when_all_have_started=max(bisect_right(t, last_first_token) for t in arrivals),
        windows=windows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    data = {p.name: summarize(p) for p in args.runs}
    assert len({v["seed"] for v in data.values()}) == 1
    print(json.dumps(data, indent=2))
