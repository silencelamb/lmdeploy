"""Two matched MTP5 diagnostics: rolling fixed I/O and a controlled prefill arrival."""

import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import aiohttp
import fixed_io
import numpy as np
from tokenizers import Tokenizer

COUNTERS = (
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "time_to_first_token_seconds_count",
    "request_prefill_time_seconds_sum",
    "request_queue_time_seconds_sum",
    "request_decode_time_seconds_sum",
    "time_to_first_token_seconds_sum",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
    "num_preemptions_total",
    "request_cache_hit_ratio_count",
    "request_cache_hit_ratio_sum",
    "prompt_tokens_cached_total",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_streams(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


async def counters(session, args, tag):
    backend = "vllm" if args.api == "completions" else "lmdeploy"

    async def fetch(i, url):
        base = url if backend == "vllm" else f"http://{urlsplit(url).hostname}:{args.monitor_port + i}"
        async with session.get(base + "/metrics") as res:
            res.raise_for_status()
            body = await res.text()
        values = {}
        for line in body.splitlines():
            if not line.startswith(backend + ":"):
                continue
            name = line.split("{", 1)[0].split()[0].split(":", 1)[1]
            if name in COUNTERS:
                values[name] = values.get(name, 0.0) + float(line.rsplit(None, 1)[-1])
        return values

    ranks = await asyncio.gather(*(fetch(i, u) for i, u in enumerate(args.urls)))
    save(args.output / (tag + "-worker-counters.json"), ranks)
    return {k: sum(r.get(k, 0) for r in ranks) for k in COUNTERS}


def difference(before, after):
    d = {k: after[k] - before[k] for k in COUNTERS}
    nd, nt, na = (d[k] for k in COUNTERS[:3])
    assert nd > 0 and 0 <= na <= nt and 0 < nt <= 5 * nd, d
    d["proposed_per_draft"] = nt / nd
    d["accepted_fraction"] = na / nt
    d["submitted_per_draft"] = 1 + na / nd
    return d


async def watch(session, args, body, i, origin, ready):
    """Mark this lane ready as soon as at least 32 output IDs have arrived."""
    url = args.urls[i % len(args.urls)]
    payload = body.copy()
    if args.api == "completions":
        payload.update(
            prompt=payload.pop("input_ids"),
            model=args.model,
            return_token_ids=True,
            add_special_tokens=False,
            stream_options={"include_usage": True},
        )
    endpoint = "/generate" if args.api == "generate" else "/v1/completions"
    begin = perf_counter()
    meta, ids, arrivals = {}, [], []
    done = False
    async with session.post(url + endpoint, json=payload) as res:
        res.raise_for_status()
        async for line in res.content:
            if not line.startswith(b"data: "):
                continue
            s = line[6:].strip()
            if s == b"[DONE]":
                done = True
                break
            event = json.loads(s)
            if args.api == "generate":
                batch = event.get("output_ids") or []
                meta = event["meta_info"]
            else:
                batch = next(iter(event.get("choices", ())), {}).get("token_ids") or []
                meta.update(event.get("usage") or {})
            if batch:
                arrivals.append([perf_counter() - begin, len(batch)])
                ids.extend(batch)
                if len(ids) >= 32:
                    ready.set()
    assert (
        done
        and len(ids) == meta["completion_tokens"] == body["max_tokens"]
        and meta["prompt_tokens"] == len(body["input_ids"])
    )
    return dict(
        index=i,
        url=url,
        started_s=begin - origin,
        finished_s=perf_counter() - origin,
        input_tokens=len(body["input_ids"]),
        output_tokens=len(ids),
        output_ids=ids,
        arrivals_s=arrivals,
        prompt_sha256=hashlib.sha256(np.asarray(body["input_ids"], dtype="<u4").tobytes()).hexdigest(),
    )


async def interference(session, args, tokenizer, repeat, inject):
    tag = f"probe32-r{repeat}-" + ("with-prefill" if inject else "alone")
    active = fixed_io.payloads(tokenizer, 1024, 512, 32, 8100 + repeat)
    long = fixed_io.payloads(tokenizer, 8192, 1, 24, 130100 + repeat)
    # Prime the 32 probe prefixes on the same workers in every arm.
    prime_start = perf_counter()
    prime = await asyncio.gather(
        *(
            fixed_io.request(session, args.urls[i % 8], dict(body, max_tokens=1), i, prime_start, args.api, args.model)
            for i, body in enumerate(active)
        )
    )
    assert all(r["error"] is None for r in prime)
    await asyncio.sleep(1)
    before = await counters(session, args, tag + "-before")
    start = perf_counter()
    ready = [asyncio.Event() for _ in active]
    pending = [asyncio.create_task(watch(session, args, body, i, start, ready[i])) for i, body in enumerate(active)]
    try:
        await asyncio.wait_for(asyncio.gather(*(flag.wait() for flag in ready)), timeout=100)
        mark = perf_counter() - start
        extra = (
            await asyncio.gather(
                *(
                    fixed_io.request(session, args.urls[i % 8], body, i, start, args.api, args.model)
                    for i, body in enumerate(long)
                )
            )
            if inject
            else []
        )
        assert all(r["error"] is None for r in extra)
        rows = await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    save_streams(args.output / (tag + "-probes.jsonl.gz"), rows)
    if extra:
        save_streams(args.output / (tag + "-prefills.jsonl.gz"), extra)
    assert max(r["started_s"] + r["arrivals_s"][0][0] for r in rows) < mark < min(r["finished_s"] for r in rows)
    if extra:
        assert max(r["started_s"] for r in extra) < min(r["finished_s"] for r in rows)
    tpot = [1000 * (r["arrivals_s"][-1][0] - r["arrivals_s"][0][0]) / (r["output_tokens"] - 1) for r in rows]
    gaps = [max((b[0] - a[0]) * 1000 for a, b in zip(r["arrivals_s"], r["arrivals_s"][1:])) for r in rows]
    await asyncio.sleep(1)
    after = await counters(session, args, tag + "-after")
    result = dict(
        tag=tag,
        injection_marker_s=mark,
        probe_count=len(rows),
        extra_count=len(extra),
        tpot_ms=fixed_io.quantiles(tpot),
        max_chunk_gap_ms_per_probe=fixed_io.quantiles(gaps),
        spec=difference(before, after),
    )
    save(args.output / (tag + "-summary.json"), result)
    print(tag, result["tpot_ms"], result["max_chunk_gap_ms_per_probe"], flush=True)


async def main(args):
    args.output.mkdir(exist_ok=False, parents=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    args.random_tokens = True
    args.placement = "least-loaded"
    args.seed = 0
    args.warmup_tokens = 8
    args.output_tokens = 256
    args.request_timeout = 240
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
        await fixed_io.warmup_case(args, tokenizer, 8192, 64, args.output)
        for repeat in (0, 1):
            tag = f"rolling-r{repeat}"
            await asyncio.sleep(1)
            before = await counters(session, args, tag + "-before")
            ok = await fixed_io.run_case(args, tokenizer, 8192, 64, repeat, args.output, 256)
            assert ok
            await asyncio.sleep(1)
            after = await counters(session, args, tag + "-after")
            save(args.output / (tag + "-counters.json"), difference(before, after))
        for repeat, order in enumerate(((False, True), (True, False))):
            for inject in order:
                await interference(session, args, tokenizer, repeat, inject)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--urls", nargs=8, required=True)
    cli.add_argument("--monitor-port", type=int, default=20980, help="First LMDeploy worker monitoring port")
    cli.add_argument("--tokenizer", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--model", default="glm53-flash")
    cli.add_argument("--api", choices=["generate", "completions"], default="generate")
    asyncio.run(main(cli.parse_args()))
