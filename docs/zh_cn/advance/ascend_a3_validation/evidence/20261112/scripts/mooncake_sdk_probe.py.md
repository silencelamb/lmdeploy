# mooncake_sdk_probe.py

原始验证文件快照；源码 SHA256：`0689f0c1a15bcf0fac4fda6b81533cc78569f24e865bdbfdf9ab9fb53c5a5788`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Owned local Store, two real NPU processes, direct buffer round trips."""
import argparse
import hashlib
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

ROOT=Path(__file__).resolve().parents[1]

def worker(role, local_device, master, inbox, outbox):
    store=None
    try:
        import torch
        import torch_npu
        torch.npu.set_device(local_device)
        from mooncake.store import MooncakeDistributedStore
        store=MooncakeDistributedStore()
        start=time.monotonic()
        ret=store.setup('127.0.0.1','P2PHANDSHAKE',64*1024**2,16*1024**2,'ascend','',master)
        print(role,'SETUP',ret,'SECONDS',time.monotonic()-start,flush=True)
        assert ret==0,ret
        outbox.put(dict(role=role,event='ready',hostname=store.get_hostname()))
        while True:
            task=inbox.get(timeout=120)
            if task is None: break
            key=task['key'];count=task['bytes']
            reference=(torch.arange(count,dtype=torch.int32)%251).to(torch.uint8)
            buffer=reference.npu() if role=='producer' else torch.zeros(count,dtype=torch.uint8,device='npu')
            torch.npu.synchronize()
            assert store.register_buffer(buffer.data_ptr(),count)==0
            start=time.monotonic()
            if role=='producer':
                ret=store.put_from(key,buffer.data_ptr(),count)
                assert ret==0,ret
            else:
                assert store.is_exist(key)==1
                ret=store.get_into(key,buffer.data_ptr(),count)
                assert ret==count,(ret,count)
            torch.npu.synchronize()
            elapsed=time.monotonic()-start
            cpu=buffer.cpu()
            assert torch.equal(cpu,reference)
            digest=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
            outbox.put(dict(role=role,event='transfer',key=key,bytes=count,ret=ret,
                seconds=elapsed,sha256=digest,exact=True))
            assert store.unregister_buffer(buffer.data_ptr())==0
    except Exception:
        outbox.put(dict(role=role,event='error',traceback=traceback.format_exc()))
        raise
    finally:
        if store is not None: store.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--port',default=55051,type=int);a=p.parse_args()
    import mooncake
    master_binary=Path(mooncake.__file__).parent/'mooncake_master'
    for port in [a.port,a.port+1]:
        with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    data=dict(version='0.3.13.post1',fabric_requested=os.environ.get('ASCEND_ENABLE_USE_FABRIC_MEM'),
              environment=os.environ['VALIDATION_ENV'],events=[],complete=False)
    def save():a.output.write_text(json.dumps(data,indent=2))
    save();ctx=mp.get_context('spawn');outbox=ctx.Queue();inputs={};children=[]
    with (ROOT/'logs'/(a.output.stem+'-master.log')).open('w') as log:
        master=subprocess.Popen([str(master_binary),f'-rpc_port={a.port}',f'-metrics_port={a.port+1}',
            '-rpc_thread_num=4','-default_kv_lease_ttl=30000','-logtostderr'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert master.poll() is None,'master exited'
                try:
                    with socket.create_connection(('127.0.0.1',a.port),timeout=.2): break
                except OSError:time.sleep(.1)
            else: raise TimeoutError('master did not listen')
            for device,role in enumerate(['producer','consumer']):
                inputs[role]=ctx.Queue()
                child=ctx.Process(target=worker,args=(role,device,f'127.0.0.1:{a.port}',inputs[role],outbox))
                child.start();children.append(child)
            def receive(event):
                result=outbox.get(timeout=120);data['events'].append(result);save()
                assert result['event']==event,result
                return result
            receive('ready');receive('ready')
            for index,count in enumerate([128*1024,2*1024**2,8*1024**2]):
                task=dict(key=f'a3-validation-{os.getpid()}-{index}',bytes=count)
                inputs['producer'].put(task);producer=receive('transfer')
                inputs['consumer'].put(task);consumer=receive('transfer')
                assert producer['sha256']==consumer['sha256']
            data['complete']=True;save()
        finally:
            for inbox in inputs.values():inbox.put(None)
            for child in children:
                child.join(timeout=10)
                if child.is_alive():child.terminate();child.join(timeout=5)
                if child.is_alive():child.kill();child.join()
            master.terminate()
            try:master.wait(timeout=10)
            except subprocess.TimeoutExpired:master.kill();master.wait()

if __name__=='__main__':main()

```
