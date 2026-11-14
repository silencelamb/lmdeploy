# http_case.py

原始验证文件快照；源码 SHA256：`e7b119e4c34910aafff28d20ed05da0d126da43e6d347b1004ece65c9cc08871`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Launch one real local API server and measure raw SSE responses reproducibly."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('--port',type=int,default=25485)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--model',default='/afs-weights/Sage-A1.5')
p.add_argument('--model-name',default='sage-a1.5')
p.add_argument('--mtp',type=int,default=3)
p.add_argument('--concurrency',type=int,nargs='+',default=[1,4,16])
a=p.parse_args()
cmd=[sys.executable,'-m','lmdeploy','serve','api_server',a.model,
     '--backend','pytorch','--device','ascend','--tp','2','--dtype','bfloat16',
     '--model-name',a.model_name,'--server-name','127.0.0.1','--server-port',str(a.port),
     '--cache-block-seq-len','128','--cache-max-entry-count','0.95','--max-batch-size','16',
     '--session-len','262144','--max-prefill-token-num','8192','--trust-remote-code',
     '--enable-prefix-caching','--prefix-cache-state-budget','16','--prefill-save-alignment','128',
     '--prefill-save-last-only','--reasoning-parser','default',
     '--tool-call-parser','qwen3coder','--language-model-only']
if a.mtp:
    cmd+=['--speculative-algorithm','qwen3_5_mtp','--speculative-num-draft-tokens',str(a.mtp)]
data=dict(command=cmd,environment=os.environ['VALIDATION_ENV'],events=[],complete=False)
def save(): a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2))
save()
server_log=ROOT/'logs'/(a.output.stem+'-server.log')
start=time.monotonic()
with server_log.open('w') as log:
    server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        url=f'http://127.0.0.1:{a.port}'
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}; see {server_log}')
            try:
                with urllib.request.urlopen(url+'/v1/models',timeout=2) as response:
                    data['models']=json.load(response)
                    break
            except Exception:
                if time.monotonic()-start>600: raise
                time.sleep(1)
        data['startup_seconds']=time.monotonic()-start
        save()

        async def request(session,prompt,tokens=128):
            payload=dict(model=a.model_name,messages=[dict(role='user',content=prompt)],
                max_tokens=tokens,temperature=0,top_k=1,seed=42,ignore_eos=True,
                stream=True,stream_options={'include_usage':True},
                chat_template_kwargs={'enable_thinking':False})
            t=time.monotonic();events=[];usage=None;text='';reasoning=''
            async with session.post(url+'/v1/chat/completions',json=payload) as res:
                if res.status!=200:
                    raise RuntimeError(f'HTTP {res.status}: {await res.text()}')
                async for raw in res.content:
                    line=raw.decode().strip()
                    if not line.startswith('data:'): continue
                    content=line[5:].strip()
                    if content=='[DONE]': break
                    obj=json.loads(content)
                    if obj.get('usage'): usage=obj['usage']
                    for choice in obj.get('choices',[]):
                        delta=choice.get('delta',{})
                        chunk=delta.get('content') or ''
                        reason=delta.get('reasoning_content') or ''
                        if chunk or reason:
                            events.append(dict(time=time.monotonic()-t,content=chunk,reasoning=reason))
                            text+=chunk;reasoning+=reason
            assert events and usage, (events,usage)
            count=usage['completion_tokens']
            assert count==tokens,(count,tokens)
            ttft=events[0]['time']
            speed=(count-1)/max(1e-9,events[-1]['time']-ttft)
            return dict(prompt=prompt,usage=usage,text=text,reasoning=reasoning,times=events,
                        elapsed=time.monotonic()-t,ttft=ttft,output_tps=speed,
                        slo_pass=ttft<=3 and speed>=40)

        async def run():
            # A slow batch can leave pooled connections idle beyond Uvicorn's
            # keep-alive timeout. Use a fresh connection for each measured request
            # on both branches; do not hide disconnects behind automatic retries.
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                              connector=aiohttp.TCPConnector(force_close=True)) as session:
                functional=await request(session,'Compute 17 * 23. Start your response with only the integer answer.',32)
                data['functional']=functional
                data['functional_answer_correct']=functional['text'].lstrip().startswith('391')
                save()
                for bs in a.concurrency:
                    for rep in range(-1,a.repeats):
                        # Different first words per C/repetition prevent cross-batch prefix reuse.
                        prompts=[f'Workload C{bs} R{rep+2} Q{i}. Explain how computers store information. '
                            + 'Discuss memory, disk, files, and cache with examples. '*12 for i in range(bs)]
                        t=time.monotonic()
                        rows=await asyncio.gather(*(request(session,s) for s in prompts))
                        elapsed=time.monotonic()-t
                        data['events'].append(dict(batch=bs,repeat=rep,warmup=rep<0,elapsed=elapsed,
                            throughput=sum(r['usage']['completion_tokens'] for r in rows)/elapsed,responses=rows))
                        save();print('HTTP BENCH',bs,rep,elapsed,flush=True)
        asyncio.run(run())
        data['complete']=True
        save()
    finally:
        server.send_signal(signal.SIGTERM)
        try: server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill();server.wait()
print('SAVED',a.output)

```
