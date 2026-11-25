"""Exercise 64 independent 64K prompts through eight actual LMDeploy HTTP
frontends."""

import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

import aiohttp

from lmdeploy.tokenizer import Tokenizer


async def generate(client, url, prompt, limit):
    started = time.perf_counter()
    payload = dict(
        input_ids=prompt, max_tokens=limit, ignore_eos=True, temperature=0, stream=True, skip_special_tokens=False
    )
    arrivals, tokens, meta, done = [], [], {}, False
    async with client.post(url + '/generate', json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f'{response.status}: {(await response.text())[:500]}')
        async for line in response.content:
            if not line.startswith(b'data: '):
                continue
            body = line[6:].strip()
            if body == b'[DONE]':
                done = True
                break
            chunk = json.loads(body)
            meta = chunk['meta_info']
            ids = chunk.get('output_ids') or []
            if ids:
                arrivals.append([time.perf_counter() - started, len(ids)])
                tokens.extend(ids)
    assert done and len(tokens) == limit and meta['prompt_tokens'] == len(prompt), (done, len(tokens), meta)
    return dict(prompt_tokens=len(prompt), started=started, tokens=tokens, arrivals=arrivals)


async def dump(directory, tag):
    (directory / ('dump-' + tag)).touch()
    for _ in range(300):
        if all((directory / f'{tag}-rank{i}.json').exists() for i in range(8)):
            return
        await asyncio.sleep(0.2)
    raise RuntimeError(f'Missing GPU timestamps for {tag}')


async def main(args):
    from benchmark_full_r11 import prompt_ids

    args.output.mkdir(exist_ok=True, parents=True)
    tokenizer = Tokenizer(args.model)
    prompts = [prompt_ids(tokenizer, rank, row) for rank in range(8) for row in range(8)]
    urls = [f'http://{args.host}:{args.port + rank}' for rank in range(8)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200), trust_env=False) as client:
        # Exercise live dummy-rank fallback without warming the B8 path.
        await generate(client, urls[0], prompts[0][:64], 4)
        waves = [('clean', 512)] if args.client_only or args.one_wave else [('clean', 512), ('profiled', 192)]
        for label, limit in waves:
            if label == 'profiled':
                count = len(json.loads((args.output / 'clean-rank0.json').read_text()))
                tmp = args.output / 'profile-at.tmp'
                tmp.write_text(str(count + 24))
                tmp.replace(args.output / 'profile-at')
            outputs = await asyncio.gather(*(generate(client, urls[i // 8], prompts[i], limit) for i in range(64)))
            with gzip.open(args.output / f'{label}-http.json.gz', 'wt') as f:
                json.dump(outputs, f)
            if not args.client_only:
                await dump(args.output, label)
            print(label, sum(len(r['tokens']) for r in outputs), flush=True)
        if args.client_only or args.one_wave:
            return
        for _ in range(300):
            if (args.output / 'worker-trace.json.gz').is_file():
                return
            await asyncio.sleep(0.2)
        raise RuntimeError('The profiled wave did not produce a CUPTI trace')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('output', type=Path)
    parser.add_argument('--client-only', action='store_true', help='Measure a plain FullR11 server without probes.')
    parser.add_argument('--one-wave', action='store_true',
                        help='Skip the separate CUPTI wave after collecting timestamps.')
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=24620)
    asyncio.run(main(parser.parse_args()))
