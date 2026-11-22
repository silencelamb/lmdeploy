"""Rebuild MTP service and prefill-interference measurements from raw token arrivals."""

import argparse
import gzip
import json
from bisect import bisect_right
from pathlib import Path
from statistics import mean

import numpy as np


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(s) for s in stream]


def pct(values):
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)}


def tpot(row):
    a = row["arrivals_s"]
    return (a[-1][0] - a[0][0]) * 1000 / (row["output_tokens"] - 1)


def biggest_gap(row):
    a = row["arrivals_s"]
    return 1000 * max(y[0] - x[0] for x, y in zip(a, a[1:]))


def steady(probes, lo, hi):
    ts = [[r["started_s"] + t for t, c in r["arrivals_s"] for _ in range(c)] for r in probes]
    # The chosen per-request intervals must be inside the interval when all lanes
    # have emitted a token and no lane has yet emitted its last token.
    if min(t[lo - 1] for t in ts) <= max(t[0] for t in ts) or max(t[hi - 1] for t in ts) >= min(t[-1] for t in ts):
        return None
    return mean((t[hi - 1] - t[lo - 1]) * 1000 / (hi - lo) for t in ts)


def spec(ds):
    rounds = sum(d["spec_decode_num_drafts_total"] for d in ds)
    proposed = sum(d["spec_decode_num_draft_tokens_total"] for d in ds)
    accepted = sum(d["spec_decode_num_accepted_tokens_total"] for d in ds)
    return dict(
        drafts=rounds,
        proposed=proposed,
        accepted=accepted,
        accepted_fraction=accepted / proposed,
        submitted_per_draft=1 + accepted / rounds,
    )


def common_decode_window(probes, lo, hi):
    ts = [[r["started_s"] + t for t, c in r["arrivals_s"] for _ in range(c)] for r in probes]
    start = max(t[lo - 1] for t in ts)
    end = min(t[hi - 1] for t in ts)
    if end <= start:
        return None
    counts = [bisect_right(t, end) - bisect_right(t, start) for t in ts]
    return dict(
        seconds=end - start,
        output_tokens=sum(counts),
        output_tokens_per_s=sum(counts) / (end - start),
        mean_tokens_per_active_user_s=sum(counts) / ((end - start) * len(probes)),
    )


def arm(run):
    assert (run / "exit_code").read_text().strip() == "0"
    f = run / "fixed"
    paths = [f / f"i8192-o256-c64-r{i}" for i in (0, 1)]
    summaries = [read(p / "summary.json") for p in paths]
    reps = [rows(p / "requests.jsonl.gz") for p in paths]
    assert all(s["success"] == s["requests"] == 256 and s["placement"] == "least-loaded" for s in summaries)
    assert all(
        row["error"] is None and sum(b for _, b in row["arrivals_s"]) == row["output_tokens"] == 256
        for rep in reps
        for row in rep
    )
    counters = [read(f / f"rolling-r{i}-counters.json") for i in (0, 1)]
    assert all(c["time_to_first_token_seconds_count"] == 256 for c in counters)
    first_waves = [sorted(rep, key=lambda r: r["started_s"])[:64] for rep in reps]
    phase = {
        k: sum(c[k + "_sum"] for c in counters) / 512
        for k in (
            "request_queue_time_seconds",
            "request_prefill_time_seconds",
            "request_decode_time_seconds",
            "time_to_first_token_seconds",
        )
    }
    rolling = dict(
        seeds=[s["seed"] for s in summaries],
        requests=512,
        output_tokens_per_s=512 * 256 / sum(s["wall_s"] for s in summaries),
        throughput_by_repeat=[s["output_tokens_per_s"] for s in summaries],
        ttft_ms=pct([r["arrivals_s"][0][0] * 1000 for rep in reps for r in rep]),
        tpot_ms=pct([tpot(r) for rep in reps for r in rep]),
        mean_phases_s=phase,
        steady_65_to_192_ms_by_repeat=[steady(rep, 65, 192) for rep in first_waves],
        spec=spec(counters),
        peak_gpu_mib=max(s["peak_gpu_mib"] for s in summaries),
    )
    interference = {}
    probes_by_tag = {}
    for tag in ("alone", "with-prefill"):
        p_list = [f / f"probe32-r{i}-{tag}-probes.jsonl.gz" for i in (0, 1)]
        pr = [rows(path) for path in p_list]
        sums = [read(f / f"probe32-r{i}-{tag}-summary.json") for i in (0, 1)]
        assert all(len(rep) == 32 and all(row["output_tokens"] == 512 for row in rep) for rep in pr)
        elapsed = []
        if tag == "with-prefill":
            for i in (0, 1):
                x = rows(f / f"probe32-r{i}-{tag}-prefills.jsonl.gz")
                assert len(x) == 24 and all(
                    r["input_tokens"] == 8192 and r["output_tokens"] == 1 and r["error"] is None for r in x
                )
                elapsed.append(max(r["finished_s"] for r in x) - min(r["started_s"] for r in x))
        result = dict(
            requests=64,
            tpot_ms=pct([tpot(r) for rep in pr for r in rep]),
            tpot_p50_by_repeat=[pct([tpot(r) for r in rep])["p50"] for rep in pr],
            max_chunk_gap_ms=pct([biggest_gap(r) for rep in pr for r in rep]),
            spec_all_live_requests=spec([s["spec"] for s in sums]),
            extra_prefill_wave_s=elapsed,
            probe_ids=[r["prompt_sha256"] for rep in pr for r in rep],
        )
        marked_gaps = []
        for rep, summary in zip(pr, sums):
            marker = summary["injection_marker_s"]
            first_end = min(r["started_s"] + r["arrivals_s"][-1][0] for r in rep)
            for r in rep:
                merged = [r["started_s"] + t for t, _ in r["arrivals_s"]]
                marked_gaps.append(1000 * max(y - x for x, y in zip(merged, merged[1:]) if marker <= y <= first_end))
        result["max_chunk_gap_while_all_live_after_marker_ms"] = pct(marked_gaps)
        if tag == "alone":
            result["steady_129_to_384_ms_by_repeat"] = [steady(rep, 129, 384) for rep in pr]
            result["common_decode_33_to_448_by_repeat"] = [common_decode_window(rep, 33, 448) for rep in pr]
        probes_by_tag[tag] = [r for rep in pr for r in rep]
        interference[tag] = result
    control = probes_by_tag["alone"]
    injected = probes_by_tag["with-prefill"]
    assert all(a["prompt_sha256"] == b["prompt_sha256"] for a, b in zip(control, injected, strict=True))
    interference["paired"] = dict(
        tpot_delta_ms=pct([tpot(b) - tpot(a) for a, b in zip(control, injected)]),
        max_chunk_gap_delta_ms=pct([biggest_gap(b) - biggest_gap(a) for a, b in zip(control, injected)]),
        identical_first_32_ids=sum(a["output_ids"][:32] == b["output_ids"][:32] for a, b in zip(control, injected)),
        identical_512_id_sequences=sum(a["output_ids"] == b["output_ids"] for a, b in zip(control, injected)),
    )
    identical = [(a, b) for a, b in zip(control, injected) if a["output_ids"] == b["output_ids"]]
    if identical:
        interference["paired"]["tpot_delta_identical_512_ids_ms"] = pct([tpot(b) - tpot(a) for a, b in identical])
    return dict(run=str(run), rolling=rolling, probe32=interference)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("lmd", type=Path)
    cli.add_argument("vllm", type=Path)
    args = cli.parse_args()
    configs = [run / "fixed" / "workload.json" for run in (args.lmd, args.vllm)]
    if any(path.exists() for path in configs):
        assert all(path.is_file() for path in configs) and read(configs[0]) == read(configs[1]), configs
    d = {"lmdeploy": arm(args.lmd), "vllm": arm(args.vllm)}
    assert d["lmdeploy"]["rolling"]["seeds"] == d["vllm"]["rolling"]["seeds"]
    hashes = d["lmdeploy"]["probe32"]["alone"]["probe_ids"]
    for eng in ("lmdeploy", "vllm"):
        for tag in ("alone", "with-prefill"):
            assert d[eng]["probe32"][tag].pop("probe_ids") == hashes
    cross = {}
    for tag in ("alone", "with-prefill"):
        lanes = {
            eng: [r for i in (0, 1) for r in rows(run / "fixed" / f"probe32-r{i}-{tag}-probes.jsonl.gz")]
            for eng, run in [("lmdeploy", args.lmd), ("vllm", args.vllm)]
        }
        pairs = list(zip(lanes["lmdeploy"], lanes["vllm"], strict=True))
        assert all(a["prompt_sha256"] == b["prompt_sha256"] for a, b in pairs)
        identical = [(a, b) for a, b in pairs if a["output_ids"] == b["output_ids"]]
        cross[tag] = dict(
            count=len(pairs),
            tpot_ratio_lmd_over_vllm=pct([tpot(a) / tpot(b) for a, b in pairs]),
            identical_512_id_sequences=len(identical),
        )
        if identical:
            cross[tag]["tpot_ratio_identical_outputs"] = pct([tpot(a) / tpot(b) for a, b in identical])
    d["lmdeploy"]["paired_against_vllm"] = cross
    print(json.dumps(d, indent=2, allow_nan=False))
