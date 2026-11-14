# model_external_cache.py

原始验证文件快照；源码 SHA256：`c73e99ae2c61e63ed01d69a101c925951eef5935b6057837edf8a7234984cf76`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Two independent TP2 engines: cold producer, fresh-cache consumer, same Store."""
import argparse
import dataclasses
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
import traceback

from fabric_probe import master_argv

ROOT=Path(__file__).resolve().parents[1]

def model_worker(role, devices, options, config_path, inbox, outbox):
    # Each engine owns a distinct visible pair and its own local Ray cluster.
    os.environ['ASCEND_RT_VISIBLE_DEVICES']=devices
    logfile=ROOT/'logs'/f"{options['label']}-{role}.log"
    with logfile.open('w',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
            from lmdeploy.messages import SpeculativeConfig, KVTransferConfig
            config=PytorchEngineConfig(device_type='ascend',dtype='bfloat16',tp=2,dp=1,ep=1,
                enable_prefix_caching=True,prefix_cache_state_budget=16,prefill_save_alignment=128,
                prefill_save_last_only=True,max_batch_size=16,session_len=262144,block_size=128,
                max_prefill_token_num=8192,cache_max_entry_count=.95,language_model_only=True,
                logprobs_mode='raw_logprobs',
                kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector',kv_role='kv_both',
                    kv_connector_extra_config={'device_backend':'ascend','mooncake_config_path':config_path,
                        'model_name':Path(options['model']).name,'cache_prefix':options['label']}))
            spec=SpeculativeConfig(method='qwen3_5_mtp',num_speculative_tokens=options['mtp']) if options['mtp'] else None
            started=time.monotonic()
            with pipeline(options['model'],backend_config=config,speculative_config=spec,
                          trust_remote_code=True,log_level='INFO') as pipe:
                outbox.put(dict(role=role,event='ready',load_seconds=time.monotonic()-started,
                    config=dataclasses.asdict(config),devices=devices,pid=os.getpid()))
                while True:
                    task=inbox.get(timeout=900)
                    if task is None:break
                    gen=GenerationConfig(max_new_tokens=32,do_sample=False,top_k=1,
                        random_seed=42,logprobs=5,ignore_eos=False)
                    start=time.monotonic();first=None
                    ids=[];scores=[];text=''
                    for response in pipe.stream_infer([task['prompt']],gen_config=gen,
                            chat_template_kwargs={'enable_thinking':False}):
                        if response.token_ids:
                            if first is None:first=time.monotonic()-start
                            ids.extend(response.token_ids)
                            scores.extend(response.logprobs or [])
                        text+=response.text
                    row=dict(token_ids=ids,logprobs=scores,text=text,
                        input_token_len=response.input_token_len,cached_tokens=response.cached_tokens,
                        finish_reason=response.finish_reason,elapsed=time.monotonic()-start,ttft=first,
                        label=task['label'],expected=task['expected'],prompt=task['prompt'],warmup=task['warmup'])
                    assert response.finish_reason not in ['error','abort'],row
                    assert task['expected'] in text,row
                    outbox.put(dict(role=role,event='response',response=row))
                modules=[n for n in sys.modules if n.split('.')[0]=='dlinfer']
                distributions=[d.metadata['Name'] for d in importlib.metadata.distributions() if 'dlinfer' in d.metadata['Name'].lower()]
                if os.environ['VALIDATION_ENV']=='native':assert not modules and not distributions
                outbox.put(dict(role=role,event='isolation',dlinfer_modules=modules,dlinfer_distributions=distributions))
            outbox.put(dict(role=role,event='closed'))
        except BaseException:
            outbox.put(dict(role=role,event='error',traceback=traceback.format_exc()))
            raise

def make_task(index, warmup=False, prefix='shared'):
    code=f'ORCHID-{8246+index}'
    filler='Archive record: the weather was mild and the road was open. '*573
    midpoint=len(filler)//2
    prompt=(f'Validation archive {prefix} item {index}.\n'+filler[:midpoint]+
        f' The unique verification code is {code}. '+filler[midpoint:]+
        ' What is the unique verification code? Reply only with the code.')
    return dict(label=f'{prefix}-{index}',prompt=prompt,expected=code,warmup=warmup)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True)
    parser.add_argument('--mtp',type=int,default=0);parser.add_argument('--label',required=True)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--port',type=int,default=55061)
    parser.add_argument('--segment-gib',type=int,default=8);parser.add_argument('--repeats',type=int,default=3)
    a=parser.parse_args();opts=vars(a).copy();opts['output']=str(opts['output'])
    for port in [a.port,a.port+1]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    data=dict(options=opts,environment=os.environ['VALIDATION_ENV'],events=[],complete=False)
    def save():a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
    config=dict(mode='embedded',metadata_server='P2PHANDSHAKE',master_server_address=f'127.0.0.1:{a.port}',
        protocol='ascend',device_name='',global_segment_size=f'{a.segment_gib}GB',local_buffer_size='4GB',enable_offload=False)
    config_path=ROOT/'configs'/f'{a.label}.json';config_path.write_text(json.dumps(config,indent=2))
    data['store_config']=config;data['effective_fabric_local_buffer_bytes']=0;save()
    import mooncake
    argv=master_argv(Path(mooncake.__file__).parent/'mooncake_master',a.port)
    data['master_argv']=argv;save()
    ctx=mp.get_context('spawn');outbox=ctx.Queue();inboxes={};children=[]
    with (ROOT/'logs'/f'{a.label}-master.log').open('w') as log:
        master=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert master.poll() is None
                try:
                    with socket.create_connection(('127.0.0.1',a.port),timeout=.2):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('master not ready')
            def receive(expected,role=None,timeout=600):
                while True:
                    item=outbox.get(timeout=timeout);data['events'].append(item);save()
                    brief={k:v for k,v in item.items() if k not in ['response','config']}
                    if 'response' in item:brief['response']={k:v for k,v in item['response'].items() if k not in ['prompt','token_ids','logprobs']}
                    print(json.dumps(brief,ensure_ascii=False),flush=True)
                    if item['event']=='isolation':continue
                    assert item['event']==expected and (role is None or item['role']==role),item
                    return item
            def start_engine(role,devices):
                inboxes[role]=ctx.Queue()
                child=ctx.Process(target=model_worker,args=(role,devices,opts,str(config_path),inboxes[role],outbox))
                child.start();children.append(child);receive('ready',role)
            tasks=[make_task(i,warmup=i==0) for i in range(a.repeats+1)]
            start_engine('producer','0,1')
            for task in tasks:
                inboxes['producer'].put(task);receive('response','producer')
            # Keep producer alive so its fabric segments and saved KV/state stay registered.
            start_engine('consumer','2,3')
            inboxes['consumer'].put(make_task(9,warmup=True,prefix='consumer-cold-warmup'))
            receive('response','consumer')
            for task in tasks:
                inboxes['consumer'].put(task);receive('response','consumer')
            for role in ['consumer','producer']:
                inboxes[role].put(None);receive('closed',role)
            data['complete']=True;save()
        except BaseException:
            data['error']=traceback.format_exc();save();raise
        finally:
            for inbox in inboxes.values():inbox.put(None)
            for child in children:
                child.join(timeout=30)
                if child.is_alive():child.terminate();child.join(timeout=10)
                if child.is_alive():child.kill();child.join(timeout=5)
            data['child_exitcodes']=[child.exitcode for child in children];save()
            master.terminate()
            try:master.wait(timeout=10)
            except subprocess.TimeoutExpired:master.kill();master.wait(timeout=5)

if __name__=='__main__':main()

```
