```python
"""Real HTTP workload with structural warmup gates and retained SSE samples."""
import argparse
import asyncio
import json
import mmap
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--mtp', type=int, default=0)
    p.add_argument('--label', required=True)
    p.add_argument('--port', type=int, default=25486)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--min-warm-sweeps', type=int, default=5)
    p.add_argument('--max-warm-sweeps', type=int, default=10)
    p.add_argument('--c1-repeats', type=int, default=25)
    p.add_argument('--c4-repeats', type=int, default=8)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    model_name = Path(a.model).name.lower()
    output = ROOT / 'results' / f'{a.label}.json'
    server_log = ROOT / 'logs' / f'{a.label}-server.log'
    assert not output.exists() and not server_log.exists()
    phase_path = Path(os.environ['WARM_CONTROL_PATH'])
    with phase_path.open('r+b') as f:
        phase = mmap.mmap(f.fileno(), 1)
    cmd = [sys.executable, str(ROOT/'scripts/draft_shape_serve.py'), 'serve', 'api_server', a.model,
           '--backend','pytorch','--device','ascend','--tp','2','--dtype','bfloat16',
           '--model-name',model_name,'--server-name','127.0.0.1','--server-port',str(a.port),
           '--cache-block-seq-len','128','--cache-max-entry-count','0.95','--max-batch-size','16',
           '--session-len','262144','--max-prefill-token-num','8192','--trust-remote-code',
           '--enable-prefix-caching','--prefix-cache-state-budget','16','--prefill-save-alignment','128',
           '--prefill-save-last-only','--reasoning-parser','default','--tool-call-parser','qwen3coder',
           '--language-model-only']
    if a.mtp:
        cmd += ['--speculative-algorithm','qwen3_5_mtp','--speculative-num-draft-tokens',str(a.mtp)]
    data = {'options': vars(a), 'server_command': cmd, 'events': [], 'phases': {}, 'complete': False,
            'metric_definition': '(completion_tokens-1)/(last_nonempty_SSE-first_nonempty_SSE)',
            'pilot': a.pilot, 'draft_coverage_supplement': True, 'performance_timings_excluded': True, 'warmup_qualified': False}
    def save():
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    def observed(pn):
        rows = []
        for file in Path(os.environ['WARM_OBSERVER_DIR']).glob('events-*.jsonl'):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if row['phase'] == pn and row['kind'] in ['new_forward_shape','triton_compile','autotune_bench','graph_capture']:
                    rows.append(row)
        return rows
    save()
    start = time.monotonic()
    with server_log.open('x') as log:
        server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{a.port}'
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                try:
                    with urllib.request.urlopen(url+'/v1/models',timeout=2) as res:
                        data['models'] = json.load(res)
                    break
                except Exception:
                    if time.monotonic()-start > 600:
                        raise
                    time.sleep(1)
            data['startup_seconds'] = time.monotonic()-start
            save()

            async def request(session, prompt):
                payload = dict(model=model_name, messages=[dict(role='user', content=prompt)],
                               max_tokens=a.tokens, temperature=0, top_k=1, seed=42, ignore_eos=True,
                               stream=True, stream_options={'include_usage': True},
                               chat_template_kwargs={'enable_thinking': False})
                began = time.monotonic()
                chunks, usage, texts, reasoning = [], None, [], []
                async with session.post(url+'/v1/chat/completions', json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(f'HTTP{response.status}: {await response.text()}')
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith('data:'):
                            continue
                        body = line[5:].strip()
                        if body == '[DONE]':
                            break
                        obj = json.loads(body)
                        if obj.get('usage'):
                            usage = obj['usage']
                        for choice in obj.get('choices', []):
                            delta = choice.get('delta', {})
                            content, reason = delta.get('content') or '', delta.get('reasoning_content') or ''
                            if content or reason:
                                chunks.append({'time': time.monotonic()-began, 'content': content, 'reasoning': reason})
                                texts.append(content)
                                reasoning.append(reason)
                elapsed = time.monotonic()-began
                assert usage and chunks and usage['completion_tokens'] == a.tokens, (usage, chunks)
                ttft = chunks[0]['time']
                speed = (a.tokens-1)/(chunks[-1]['time']-ttft)
                return {'prompt': prompt, 'text': ''.join(texts), 'reasoning': ''.join(reasoning),
                        'usage': usage, 'events': chunks, 'ttft': ttft, 'elapsed': elapsed,
                        'output_tps': speed, 'tpot': 1/speed}

            async def run_batch(session, bs, rep, kind):
                pn = len(data['phases'])+1
                assert pn < 256
                data['phases'][str(pn)] = {'kind': kind, 'batch': bs, 'rep': rep}
                phase[0] = pn
                # This tokenizer splits digits: keep R/Q single-digit as in the
                # historical171-token input while using disjoint cache-miss IDs.
                first = rep*bs if kind=='warmup' else 96 if kind=='qualify' else 40+rep*bs
                assert first+bs <= 100
                prompts = [f'Workload C{bs} R{(first+i)//10} Q{(first+i)%10}. Explain how computers store information. '
                           + 'Discuss memory, disk, files, and cache with examples. '*12 for i in range(bs)]
                began = time.monotonic()
                rows = await asyncio.gather(*(request(session, text) for text in prompts))
                elapsed = time.monotonic()-began
                cold = observed(pn)
                event = {'kind': kind, 'phase': pn, 'batch': bs, 'rep': rep, 'responses': rows,
                         'elapsed': elapsed, 'throughput': bs*a.tokens/elapsed,
                         'cold_events': cold, 'qualified': not cold}
                data['events'].append(event)
                save()
                print('BATCH',kind,bs,rep,'tps',round(event['throughput'],3),'cold',len(cold),flush=True)
                return event

            async def bench():
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                                  connector=aiohttp.TCPConnector(force_close=True)) as session:
                    quiet = 0
                    for sweep in range(a.max_warm_sweeps):
                        events = [await run_batch(session, bs, sweep, 'warmup') for bs in [1,2,3,4]]
                        quiet = quiet+1 if all(e['qualified'] for e in events) else 0
                        if sweep+1 >= a.min_warm_sweeps and quiet >= 2:
                            data['warmup_qualified'] = True
                            break
                    save()
                    if not data['warmup_qualified']:
                        raise RuntimeError('Warmup did not reach two quiet sweeps within the fixed limit')
                    for bs in [1,4]:
                        await run_batch(session, bs, 0, 'qualify')
                    for rep in range(max(a.c1_repeats, a.c4_repeats)):
                        order = [1,4] if rep%2 == 0 else [4,1]
                        for bs in order:
                            if rep < (a.c1_repeats if bs==1 else a.c4_repeats):
                                await run_batch(session, bs, rep, 'measure')
            asyncio.run(bench())
            data['measurement_qualified'] = all(e['qualified'] for e in data['events'] if e['kind']=='measure')
            data['complete'] = True
            save()
        finally:
            phase[0] = 255
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            data['server_returncode'] = server.returncode
            data['worker_summaries'] = [json.loads(p.read_text()) for p in Path(os.environ['WARM_OBSERVER_DIR']).glob('summary-*.json')]
            data['driver_summaries'] = [json.loads(p.read_text()) for p in Path(os.environ['WARM_OBSERVER_DIR']).glob('driver-*.json')]
            data['token_traces'] = [json.loads(line) for p in Path(os.environ['WARM_OBSERVER_DIR']).glob('tokens-*.jsonl') for line in p.read_text().splitlines()]
            save()
            phase.close()


if __name__ == '__main__':
    main()
```
