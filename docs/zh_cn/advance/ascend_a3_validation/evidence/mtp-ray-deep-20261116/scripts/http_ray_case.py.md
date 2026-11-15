# http_ray_case.py

```python
"""Run a real local streaming API and correlate SSE times with RPC probes."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT = Path(os.environ['VALIDATION_ROOT'])
p = argparse.ArgumentParser()
p.add_argument('--output', required=True, type=Path)
p.add_argument('--port', type=int, default=25516)
p.add_argument('--repeats', type=int, default=6)
p.add_argument('--tokens', type=int, default=256)
p.add_argument('--concurrency', type=int, nargs='+', default=[1, 4])
a = p.parse_args()
assert not a.output.exists(), a.output
with socket.socket() as probe:
    probe.bind(('127.0.0.1', a.port))
cmd = [sys.executable, str(ROOT/'scripts/serve_ray_trace.py'), 'serve', 'api_server',
       '/afs-weights/Agents-A1', '--backend', 'pytorch', '--device', 'ascend',
       '--tp', '2', '--dtype', 'bfloat16', '--model-name', 'agents-a1',
       '--server-name', '127.0.0.1', '--server-port', str(a.port),
       '--cache-block-seq-len', '128', '--cache-max-entry-count', '0.5',
       '--max-batch-size', '8', '--session-len', '8192',
       '--max-prefill-token-num', '256', '--trust-remote-code', '--language-model-only']
data = dict(command=cmd, options=vars(a) | {'output': str(a.output)},
            events=[], complete=False, server_log=str(ROOT/'logs'/f'{a.output.stem}-server.log'))


def save():
    a.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


save()
started = time.monotonic()
with open(data['server_log'], 'x') as log:
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    data['server_pid'] = server.pid
    try:
        url = f'http://127.0.0.1:{a.port}'
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            try:
                with urllib.request.urlopen(url+'/v1/models', timeout=2) as response:
                    data['models'] = json.load(response)
                break
            except Exception:
                if time.monotonic()-started > 600:
                    raise
                time.sleep(1)
        data['startup_seconds'] = time.monotonic()-started
        save()

        async def request(session, prompt, tokens):
            payload = dict(model='agents-a1', messages=[dict(role='user', content=prompt)],
                           max_tokens=tokens, temperature=0, top_k=1, seed=42,
                           ignore_eos=True, stream=True, stream_options={'include_usage': True},
                           chat_template_kwargs={'enable_thinking': False})
            began = time.monotonic_ns()
            events = []
            usage = None
            async with session.post(url+'/v1/chat/completions', json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f'HTTP {response.status}: {await response.text()}')
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    content = line[5:].strip()
                    if content == '[DONE]':
                        break
                    obj = json.loads(content)
                    if obj.get('usage'):
                        usage = obj['usage']
                    for choice in obj.get('choices', []):
                        delta = choice.get('delta', {})
                        text = delta.get('content') or ''
                        reason = delta.get('reasoning_content') or ''
                        if text or reason:
                            events.append(dict(t_ns=time.monotonic_ns(), content=text, reasoning=reason))
            assert events and usage and usage['completion_tokens'] == tokens, usage
            return dict(prompt=prompt, start_ns=began, end_ns=time.monotonic_ns(),
                        usage=usage, events=events)

        async def run():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                    connector=aiohttp.TCPConnector(force_close=True)) as session:
                for concurrency in a.concurrency:
                    for rep in range(-1, a.repeats):
                        prompts = [f'Workload C{concurrency} R{rep+2} Q{i}. Explain how computers store information. '
                                   + 'Discuss memory, disk, files, and cache with examples. '*12
                                   for i in range(concurrency)]
                        began = time.monotonic_ns()
                        rows = await asyncio.gather(*(request(session, prompt, a.tokens) for prompt in prompts))
                        ended = time.monotonic_ns()
                        data['events'].append(dict(concurrency=concurrency, repeat=rep,
                                                  warmup=rep<0, start_ns=began, end_ns=ended, responses=rows))
                        save()
                        print('HTTP', concurrency, rep, round((ended-began)/1e9, 3), flush=True)
        asyncio.run(run())
        data['complete'] = True
        save()
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        data['server_returncode'] = server.returncode
        save()
print('SAVED', a.output)
```
