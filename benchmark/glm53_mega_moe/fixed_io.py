"""Fixed-token LMDeploy serving benchmark, with DP-balanced placement."""

import argparse
import asyncio
import gzip
import json
import random
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
from tokenizers import Tokenizer


def quantiles(values):
    return {f'p{q}': float(np.percentile(values, q)) for q in (50, 90, 95, 99)} if len(values) else {}


def payloads(tokenizer, length, output_length, total, seed):
    # Use normal text tokens with per-request prefixes to keep Fixed I/O cold
    # even when the same server is also running prefix-cache trace benchmarks.
    source = ('Each request has an independent decoding state. A GPU inference worker '
              'uses attention over cached context, then routes tokens to distributed experts. '
              'We measure first-token latency, inter-token latency, throughput, and memory.\n'
              '每个请求独立维护状态。GPU 推理执行注意力计算、专家路由和跨卡通信。\n')
    words = tokenizer.encode(source, add_special_tokens=False).ids
    tail = tokenizer.encode('\nDescribe how these measurements help operate a serving system.\n',
                            add_special_tokens=False).ids
    rng = random.Random(seed)
    bodies = []
    for index in range(total):
        prefix = rng.choices(words, k=24)
        phase = (seed + index) % len(words)
        context = (words[phase:] + words * (length // len(words) + 1))[:length - len(prefix) - len(tail)]
        bodies.append(dict(input_ids=prefix + context + tail, max_tokens=output_length, stream=True, ignore_eos=True,
                           skip_special_tokens=False, temperature=0))
    return bodies


def random_payloads(tokenizer, length, output_length, total, seed):
    added = tokenizer.get_added_tokens_decoder()
    special = {token for token, value in added.items() if value.special}
    vocab = np.array(sorted(set(tokenizer.get_vocab(with_added_tokens=False).values()) - special))
    rng = np.random.default_rng(seed)
    offsets = rng.integers(len(vocab), size=total)
    positions = np.arange(length)
    return [dict(input_ids=vocab[(offset + i + positions) % len(vocab)].tolist(),
                 max_tokens=output_length, stream=True, ignore_eos=True, skip_special_tokens=False, temperature=0)
            for i, offset in enumerate(offsets)]


async def request(session, url, body, index, origin, api='generate', model='glm53-flash'):
    started = time.perf_counter()
    arrival = []
    output = []
    meta = {}
    done = False
    error = None
    if api == 'completions':
        endpoint = '/v1/completions'
        payload = {key: value for key, value in body.items() if key != 'input_ids'}
        payload.update(model=model, prompt=body['input_ids'], return_token_ids=True, add_special_tokens=False,
                       stream_options={'include_usage': True})
    else:
        endpoint, payload = '/generate', body
    try:
        async with session.post(f'{url}{endpoint}', json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status} {await response.text()}')
            async for line in response.content:
                if not line.startswith(b'data: '):
                    continue
                message = line[6:].strip()
                if message == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(message)
                if 'error' in chunk:
                    raise RuntimeError(str(chunk['error']))
                if api == 'completions':
                    meta.update(chunk.get('usage') or {})
                    choice = next(iter(chunk.get('choices', ())), {})
                    if choice.get('finish_reason'):
                        meta['finish_reason'] = choice['finish_reason']
                    ids = choice.get('token_ids') or []
                else:
                    meta = chunk['meta_info']
                    ids = chunk.get('output_ids') or []
                if ids:
                    arrival.append((time.perf_counter() - started, len(ids)))
                    output.extend(ids)
        expected = len(body['input_ids']), body['max_tokens']
        actual = meta.get('prompt_tokens'), meta.get('completion_tokens')
        if not done or expected != actual or len(output) != expected[1]:
            raise RuntimeError(f'Expected {expected} tokens, got {actual}, {len(output)} streamed and done={done}.')
    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError, KeyError) as exc:
        error = str(exc) or type(exc).__name__
    return dict(index=index, url=url, started_s=started - origin, finished_s=time.perf_counter() - origin,
                input_tokens=meta.get('prompt_tokens'), output_tokens=len(output),
                finish_reason=meta.get('finish_reason'), output_ids=output, arrivals_s=arrival, error=error)


async def run_case(args, tokenizer, context, concurrency, repeat, output, request_count=None):
    case = output / f'i{context}-o{args.output_tokens}-c{concurrency}-r{repeat}'
    case.mkdir()
    seed = args.seed + 100003 * context + 1009 * concurrency + repeat
    total = concurrency * args.waves if request_count is None else request_count
    make_payloads = random_payloads if args.random_tokens else payloads
    bodies = make_payloads(tokenizer, context, args.output_tokens, total, seed)
    api_args = args.api, args.model
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    origin = time.perf_counter()
    rows = []
    with (case / 'gpu.csv').open('w') as gpu_csv:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw',
                                    '--format=csv,noheader,nounits', '-lms', '1000'], stdout=gpu_csv)
        try:
            connector = aiohttp.TCPConnector(limit=concurrency)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as conn:
                if request_count is None:
                    for start in range(0, total, concurrency):
                        rows.extend(await asyncio.gather(*[
                            request(conn, args.urls[i % len(args.urls)], bodies[i], i, origin, *api_args)
                            for i in range(start, start + concurrency)
                        ]))
                else:
                    semaphore = asyncio.Semaphore(concurrency)
                    pending = [0] * len(args.urls)

                    async def submit(i):
                        async with semaphore:
                            preferred = i % len(args.urls)
                            worker = preferred
                            if args.placement == 'least-loaded':
                                worker = min(range(len(pending)),
                                             key=lambda j: (pending[j], (j - preferred) % len(pending)))
                            pending[worker] += 1
                            try:
                                return await request(conn, args.urls[worker], bodies[i], i, origin, *api_args)
                            finally:
                                pending[worker] -= 1

                    rows = await asyncio.gather(*(submit(i) for i in range(total)))
        finally:
            monitor.terminate()
            monitor.wait()
    with gzip.open(case / 'requests.jsonl.gz', 'wt', encoding='utf-8') as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + '\n')

    good = [r for r in rows if r['error'] is None]
    wall = max(r['finished_s'] for r in rows) - min(r['started_s'] for r in rows)
    ttft = [r['arrivals_s'][0][0] * 1000 for r in good]
    tpot = [(r['arrivals_s'][-1][0] - r['arrivals_s'][0][0]) * 1000 / (r['output_tokens'] - 1)
            for r in good if r['output_tokens'] > 1]
    itl = []
    for row in good:
        for (previous, _), (now, count) in zip(row['arrivals_s'], row['arrivals_s'][1:]):
            itl.extend([(now - previous) * 1000 / count] * count)
    e2e = [(r['finished_s'] - r['started_s']) * 1000 for r in good]
    summary = dict(context=context, output_tokens=args.output_tokens, concurrency=concurrency, repeat=repeat, seed=seed,
                   wall_s=wall, requests=total, success=len(good), placement=args.placement,
                   ttft_ms=quantiles(ttft), tpot_ms=quantiles(tpot),
                   itl_ms=quantiles(itl), mean_itl_ms=float(np.mean(itl)) if itl else None,
                   end_to_end_ms=quantiles(e2e), requests_per_s=len(good) / wall,
                   input_tokens_per_s=sum(r['input_tokens'] for r in good) / wall,
                   output_tokens_per_s=sum(r['output_tokens'] for r in good) / wall,
                   peak_gpu_mib=max((int(line.split(',')[2]) for line in (case / 'gpu.csv').read_text().splitlines()),
                                    default=None))
    (case / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return len(good) == total


async def warmup_case(args, tokenizer, context, concurrency, output):
    make_payloads = random_payloads if args.random_tokens else payloads
    seed = args.seed - context - concurrency
    if args.random_tokens:
        seed %= 1 << 32
    bodies = make_payloads(tokenizer, context, args.warmup_tokens, concurrency, seed)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as session:
        origin = time.perf_counter()
        rows = await asyncio.gather(*[
            request(session, args.urls[i % len(args.urls)], body, i, origin, args.api, args.model)
            for i, body in enumerate(bodies)
        ])
    (output / f'warmup-i{context}-c{concurrency}.json').write_text(json.dumps(rows) + '\n')
    if any(row['error'] is not None for row in rows):
        raise RuntimeError(f'Warmup failed for context={context}, concurrency={concurrency}.')


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / 'tokenizer.json'))
    ok = True
    cases = args.case or [(context, args.output_tokens, concurrency, None)
                          for context in args.contexts for concurrency in args.concurrencies]
    for context, output_tokens, concurrency, count in cases:
        args.output_tokens = output_tokens
        if args.warmup_tokens:
            await warmup_case(args, tokenizer, context, concurrency, output)
        for repeat in range(args.repeats):
            ok &= await run_case(args, tokenizer, context, concurrency, repeat, output, count)
    if not ok:
        raise SystemExit('Some Fixed I/O requests failed length or stream checks.')


def parse_case(text):
    try:
        context, output, concurrency, count = map(int, text.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected INPUT:OUTPUT:CONCURRENCY:REQUESTS.') from exc
    if min(context, output, concurrency) <= 0 or count < concurrency:
        raise argparse.ArgumentTypeError('Lengths and concurrency must be positive, with enough requests.')
    return context, output, concurrency, count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls', required=True, nargs='+')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--contexts', nargs='+', type=int, default=[2048, 8192, 32768, 65536])
    parser.add_argument('--concurrencies', nargs='+', type=int, default=[8, 32, 64])
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup-tokens', type=int, default=8)
    parser.add_argument('--waves', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--case', type=parse_case, action='append',
                        help='INPUT:OUTPUT:CONCURRENCY:REQUESTS with rolling concurrency.')
    parser.add_argument('--random-tokens', action='store_true')
    parser.add_argument('--placement', choices=('round-robin', 'least-loaded'), default='round-robin')
    parser.add_argument('--api', choices=('generate', 'completions'), default='generate')
    parser.add_argument('--model', default='glm53-flash')
    asyncio.run(main(parser.parse_args()))
