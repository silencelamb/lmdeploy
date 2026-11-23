"""Measure real MTP commit lengths and stream intervals while every DP rank has the same live batch."""
import argparse
import asyncio
import gzip
import json
from pathlib import Path
from time import perf_counter

import aiohttp
from tokenizers import Tokenizer

import fixed_io
import mtp_interference


def steady_window(rows):
    """Keep entire adjacent stream intervals inside the common steady generation window."""
    timelines = []
    for row in rows:
        position = 0
        timeline = []
        for arrival, length in row['arrivals_s']:
            position += length
            timeline.append((row['started_s'] + arrival, length, position))
        timelines.append(timeline)
    start = max(next(t for t, _, pos in line if pos >= 32) for line in timelines)
    end = min(next(t for t, _, pos in line if pos >= 480) for line in timelines)
    elapsed_ms = 0.
    intervals = tokens = 0
    for line in timelines:
        for (previous, _, _), (current, length, _) in zip(line, line[1:]):
            if start <= previous < current <= end:
                elapsed_ms += (current - previous) * 1000
                intervals += 1
                tokens += length
    if not intervals:
        raise ValueError('No common steady window with all requests active.')
    return dict(request_rounds=intervals, delivered_tokens=tokens, sum_request_ms=elapsed_ms,
                ms_per_round=elapsed_ms / intervals, submitted_per_round=tokens / intervals,
                request_ms_per_token=elapsed_ms / tokens)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async def wave(bodies):
            start = perf_counter()
            rows = await asyncio.gather(*(fixed_io.request(session, args.urls[i % len(args.urls)], body, i, start)
                                          for i, body in enumerate(bodies)))
            errors = [r['error'] for r in rows if r['error']]
            if errors:
                raise RuntimeError(errors)
            return rows

        for repeat in range(args.repeats):
            for batch in args.batch_per_rank:
                tag = f'b{batch}-r{repeat}'
                prompts = fixed_io.payloads(tokenizer, 1024, 512, batch, args.seed + repeat)
                bodies = [prompt for prompt in prompts for _ in args.urls]
                await wave([dict(body, max_tokens=1) for body in bodies])
                await asyncio.sleep(.5)
                before = await mtp_interference.counters(session, args, tag + '-before')
                rows = await wave(bodies)
                with gzip.open(args.output / (tag + '.jsonl.gz'), 'wt') as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + '\n')
                after = await mtp_interference.counters(session, args, tag + '-after')
                summary = steady_window(rows)
                if args.draft_depth:
                    delta = mtp_interference.difference(before, after, args.draft_depth)
                    chunks = sum(len(row['arrivals_s']) - 1 for row in rows)
                    if chunks != delta['spec_decode_num_drafts_total']:
                        raise ValueError(f'{tag}: {chunks} decode chunks, {delta["spec_decode_num_drafts_total"]} drafts')
                    summary['worker_submitted_per_draft'] = delta['submitted_per_draft']
                elif any(n != 1 for r in rows for _, n in r['arrivals_s']):
                    raise ValueError('Expected a single delivered token per MTP0 step.')
                (args.output / (tag + '-summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
                print(tag, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls', nargs=8, required=True)
    parser.add_argument('--monitor-port', type=int, default=20980)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft-depth', type=int, choices=(0, 3, 5), required=True)
    parser.add_argument('--batch-per-rank', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--seed', type=int, default=8100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.set_defaults(api='generate', model='glm53-flash')
    opts = parser.parse_args()
    if min(*opts.batch_per_rank, opts.repeats) < 1:
        parser.error('Batch sizes and repeats must be positive.')
    asyncio.run(run(opts))
