"""Run the MTP0 scheduling control with client streams and eight worker phase histograms."""

import argparse
import asyncio
import json
from pathlib import Path
from time import time
from urllib.parse import urlsplit

import aiohttp
from fixed_io import run_case, warmup_case
from tokenizers import Tokenizer

PHASES = ('time_to_first_token_seconds', 'request_queue_time_seconds', 'request_prefill_time_seconds',
          'request_decode_time_seconds', 'e2e_request_latency_seconds')


async def snapshot(client, args, name):
    async def fetch(url):
        async with client.get(url) as response:
            response.raise_for_status()
            return await response.text()

    workers = [f'{urlsplit(url).scheme}://{urlsplit(url).hostname}:{args.monitor_port + rank}/metrics'
               for rank, url in enumerate(args.urls)]
    responses = await asyncio.gather(*(fetch(url) for url in workers))
    counters = dict.fromkeys((f'lmdeploy:{phase}_{kind}' for phase in PHASES for kind in ('sum', 'count')), 0.)
    for rank, body in enumerate(responses):
        (args.output / f'{name}-rank{rank}.prom').write_text(body)
        for line in body.splitlines():
            if line and not line.startswith('#'):
                key = line.split('{', 1)[0].split(maxsplit=1)[0]
                if key in counters:
                    counters[key] += float(line.rsplit(maxsplit=1)[-1])
    return counters


async def main(args):
    assert len(args.urls) == 8
    args.output.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / 'tokenizer.json'))
    args.api, args.model = 'generate', 'glm53-flash'
    args.warmup_tokens, args.request_timeout, args.seed = 8, 600, 0
    args.random_tokens, args.placement = True, 'least-loaded'
    summaries = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as client:
        for output, concurrency, count in ((1, 8, 64), (256, 64, 256)):
            args.output_tokens = output
            name = f'i8192-o{output}-c{concurrency}'
            await warmup_case(args, tokenizer, 8192, concurrency, args.output)
            before = await snapshot(client, args, f'{name}-before')
            started = time()
            valid = await run_case(args, tokenizer, 8192, concurrency, 0, args.output, count)
            finished = time()
            if output == 256:
                (args.output / 'rolling-clock.json').write_text(json.dumps([[started, finished]]) + '\n')
            await asyncio.sleep(1)
            after = await snapshot(client, args, f'{name}-after')
            if not valid:
                raise RuntimeError(f'{name} produced an invalid stream.')
            phases = {}
            for phase in PHASES:
                stem = f'lmdeploy:{phase}'
                observations = after[stem + '_count'] - before[stem + '_count']
                if observations != count:
                    raise ValueError(f'{name} has {observations} observations of {phase}, expected {count}.')
                total = after[stem + '_sum'] - before[stem + '_sum']
                phases[phase] = dict(count=count, sum_s=total, mean_s=total / count)
            summaries.append(dict(shape=name, requests=count, stream_valid=valid, phases=phases))
    (args.output / 'phase-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls', nargs='+', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monitor-port', type=int, default=20980)
    asyncio.run(main(parser.parse_args()))
