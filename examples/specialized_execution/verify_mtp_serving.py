"""Compare MTP3 greedy serving runs on a real GLM-5.3 checkpoint."""

import argparse
import asyncio
import hashlib
import json
import re
import socket
import subprocess
import time
from pathlib import Path

import aiohttp


async def client(server, output, model, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(('8.8.8.8', 0))
        ip = sock.getsockname()[0]
    urls = [f'http://{ip}:{port + rank}' for rank in range(8)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240), trust_env=False) as session:

        async def ready(url):
            try:
                async with session.get(url + '/v1/models', timeout=aiohttp.ClientTimeout(total=2)) as response:
                    return response.status == 200 and (await response.json())['data'][0]['id'] == model
            except (OSError, aiohttp.ClientError, TimeoutError, IndexError, KeyError):
                return False

        deadline = time.monotonic() + 700
        while not all(await asyncio.gather(*(ready(url) for url in urls))):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'Worker readiness failed, process status {server.poll()}')
            await asyncio.sleep(2)

        async def generate(rank, request, length):
            # The two rows have equal prefill lengths and different token IDs on every rank.
            prompt = [128 + (37 * request + j * 13) % 2000 for j in range(64)]
            payload = dict(
                input_ids=prompt, max_tokens=length, temperature=1.0, top_k=1, top_p=1.0, ignore_eos=True, stream=False
            )
            async with session.post(urls[rank] + '/generate', json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f'{response.status}: {(await response.text())[:300]}')
                obj = await response.json()
                ids, meta = obj['output_ids'], obj['meta_info']
                assert len(ids) == meta['completion_tokens'] == length, meta
                assert len(prompt) == meta['prompt_tokens'], meta
                return dict(request=request, rank=rank, ids=ids)

        results = [await generate(0, 0, 8)]
        results.extend(await asyncio.gather(*(generate(k // 2, k + 1, 96) for k in range(16))))
        (output / 'tokens.json').write_text(json.dumps(results) + '\n')
        metrics = []
        for url in urls:
            async with session.get(url + '/metrics') as response:
                metrics.append(
                    '\n'.join(
                        line
                        for line in (await response.text()).splitlines()
                        if 'spec_decode' in line and not line.startswith('#')
                    )
                )
        (output / 'spec-metrics.txt').write_text('\n'.join(metrics) + '\n')
        return results


def live_shadow_summary(log):
    prefix = 'MTP3_LIVE_SHADOW '
    records = [json.loads(line.split(prefix, 1)[1]) for line in log.splitlines() if prefix in line]
    ranks = sorted({row['rank'] for row in records})
    assert ranks == list(range(8)), f'Missing same-input projection checks: {ranks}'
    assert all(row['layers'] == 34 and row['values'] == 34 * 8 * 4096 for row in records)
    return dict(
        ranks=ranks,
        worker_forwards=len(records),
        projection_outputs=sum(row['values'] for row in records),
        different=sum(row['different'] for row in records),
        max_abs=max(row['max_abs'] for row in records),
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        'lmdeploy',
        'serve',
        'api_server',
        args.model,
        '--backend',
        'pytorch',
        '--distributed-executor-backend',
        'ray',
        '--language-model-only',
        '--trust-remote-code',
        '--tp',
        '1',
        '--dp',
        '8',
        '--ep',
        '8',
        '--max-batch-size',
        '2',
        '--session-len',
        '256',
        '--num-gpu-blocks',
        '128',
        '--cache-block-seq-len',
        '64',
        '--generation-config',
        'lmdeploy',
        '--max-prefill-token-num',
        '64',
        '--cudagraph-capture-batch-sizes',
        '1',
        '2',
        '--speculative-algorithm',
        'deepseek_mtp',
        '--speculative-num-draft-tokens',
        '3',
        '--server-port',
        str(args.port),
        '--model-name',
        'glm53-mtp',
        '--log-level',
        'DEBUG',
    ]
    if args.provider:
        command.extend(['--specialized-executor', args.provider])
    if args.eager:
        command.append('--eager-mode')
    (args.output / 'command.json').write_text(json.dumps(command) + '\n')
    with (args.output / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            results = asyncio.run(client(proc, args.output, 'glm53-mtp', args.port))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    data = (args.output / 'tokens.json').read_bytes()
    summary = dict(
        requests=len(results),
        output_ids=sum(len(row['ids']) for row in results),
        tokens_sha256=hashlib.sha256(data).hexdigest(),
    )
    log = (args.output / 'server.log').read_text(errors='replace')
    if args.provider == 'glm53_r11:ShadowMTP3Projection':
        summary['shadow'] = live_shadow_summary(log)
    if args.provider == 'glm53_r11:MTP3Projection' and not args.eager:
        replay = [line for line in log.splitlines() if 'Replaying specialized decode graph kda-mtp3-verify-b2' in line]
        summary['graph_replay_logs'] = len(replay)
        summary['replay_workers'] = len(
            {m.group(1) for line in replay if (m := re.search(r'RayWorkerWrapper pid=(\d+)', line))}
        )
        assert summary['replay_workers'] == 8, summary
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert [row['request'] for row in reference] == [row['request'] for row in results]
        summary['matching_requests'] = sum(a == b for a, b in zip(results, reference))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if args.reference:
        assert summary['matching_requests'] == len(results), 'Output token mismatch against the independent run'
    if 'shadow' in summary:
        assert summary['shadow']['different'] == 0, 'Live native projections differ from stock on the same activations'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('output', type=Path)
    parser.add_argument('--provider')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--port', type=int, default=24910)
    main(parser.parse_args())
