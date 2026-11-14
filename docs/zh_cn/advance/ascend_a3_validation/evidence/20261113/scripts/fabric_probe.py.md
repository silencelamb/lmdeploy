# fabric_probe.py

原始验证文件快照；源码 SHA256：`0165eb3121ac5c096ea54a2b38ba0a84c111b4987a1cecf1a2b8095d209b983d`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Exercise the pinned PR backend with actual NPU buffers in two processes."""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def worker(role, device, master, segment_gib, io_api, inbox, outbox):
    store=None
    backend=None
    setup_ok=False
    try:
        import torch
        import torch_npu
        torch.npu.set_device(device)
        torch.ones(1,device='npu').cpu()
        from mooncake.store import MooncakeDistributedStore
        from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreConfig
        from lmdeploy.pytorch.kv_connector.mooncake.store.transfer_backend import AscendTransferBackend
        from lmdeploy.pytorch.kv_connector.mooncake.store.worker import _get_local_hostname
        backend=AscendTransferBackend()
        assert backend.use_fabric_mem
        config=MooncakeStoreConfig(metadata_server='P2PHANDSHAKE',master_server_address=master,
            protocol='ascend',device_name='',global_segment_size=segment_gib*1024**3,
            local_buffer_size=4*1024**3)
        hostname=_get_local_hostname()
        outbox.put(dict(role=role,event='setup-start',pid=os.getpid(),device=device,
            hostname=hostname,segment_bytes=config.global_segment_size,effective_local_buffer=0,
            external_engine=False,client_buffer_registration=False))
        store=MooncakeDistributedStore()
        start=time.monotonic()
        ret=backend.setup_store(store,config,hostname)
        assert ret==0,ret
        setup_ok=True
        outbox.put(dict(role=role,event='ready',ret=ret,seconds=time.monotonic()-start))
        while True:
            task=inbox.get(timeout=90)
            if task is None:break
            n=task['bytes'];offset=task['offset']
            expected=((torch.arange(n,dtype=torch.int32)+offset)%251).to(torch.uint8)
            buffer=expected.npu() if task['op']=='put' else torch.full((n,),255,dtype=torch.uint8,device='npu')
            torch.npu.synchronize()
            start=time.monotonic()
            sizes=[n//4,n//2,n//4]
            addresses=[buffer.data_ptr(),buffer.data_ptr()+sizes[0],buffer.data_ptr()+sizes[0]+sizes[1]]
            if task['op']=='put':
                if io_api=='batch':
                    ret=list(store.batch_put_from_multi_buffers([task['key']],[addresses],[sizes]))
                    assert ret==[0],ret
                else:
                    ret=store.put_from(task['key'],buffer.data_ptr(),n)
                    assert ret==0,ret
            else:
                assert store.is_exist(task['key'])==1
                if io_api=='batch':
                    ret=list(store.batch_get_into_multi_buffers([task['key']],[addresses],[sizes]))
                    assert ret==[n],(ret,n)
                else:
                    ret=store.get_into(task['key'],buffer.data_ptr(),n)
                    assert ret==n,(ret,n)
            torch.npu.synchronize()
            actual=buffer.cpu()
            assert torch.equal(actual,expected)
            outbox.put(dict(role=role,event=task['op'],key=task['key'],bytes=n,ret=ret,
                seconds=time.monotonic()-start,sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),exact=True))
    except BaseException:
        outbox.put(dict(role=role,event='error',traceback=traceback.format_exc()))
        raise
    finally:
        if setup_ok:
            start=time.monotonic()
            store.close()
            outbox.put(dict(role=role,event='closed',seconds=time.monotonic()-start))
        if backend is not None:backend.close()

def master_argv(binary,port):
    return [str(binary),f'--rpc_port={port}',f'--metrics_port={port+1}',
        '--rpc_thread_num=4','--eviction_high_watermark_ratio=0.9','--eviction_ratio=0.1',
        '--default_kv_lease_ttl=30000','--enable_offload=false','--client_ttl=120','--logtostderr']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--port',type=int,default=55061);p.add_argument('--segment-gib',type=int,default=1)
    p.add_argument('--io-api',choices=['batch','scalar'],default='batch')
    a=p.parse_args()
    import mooncake
    version=importlib.metadata.version('mooncake-transfer-engine-npu')
    assert version=='0.3.13.post1',version
    for port in [a.port,a.port+1]:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    data=dict(version=version,options=dict(port=a.port,segment_gib=a.segment_gib,io_api=a.io_api),
        environment=os.environ['VALIDATION_ENV'],events=[],complete=False)
    def save():a.output.write_text(json.dumps(data,indent=2))
    save();ctx=mp.get_context('spawn');outbox=ctx.Queue();inboxes={};children=[]
    argv=master_argv(Path(mooncake.__file__).parent/'mooncake_master',a.port)
    data['master_argv']=argv;save()
    with (ROOT/'logs'/(a.output.stem+'-master.log')).open('w') as log:
        master=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert master.poll() is None,'master exited'
                try:
                    with socket.create_connection(('127.0.0.1',a.port),timeout=.2):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('master not ready')
            for dev,role in enumerate(['a','b']):
                inboxes[role]=ctx.Queue()
                child=ctx.Process(target=worker,args=(role,dev,f'127.0.0.1:{a.port}',a.segment_gib,a.io_api,inboxes[role],outbox))
                child.start();children.append(child)
            def receive(event):
                while True:
                    result=outbox.get(timeout=90);data['events'].append(result);save()
                    print(json.dumps(result),flush=True)
                    if result['event']=='setup-start':continue
                    assert result['event']==event,result
                    return result
            receive('ready');receive('ready')
            for direction in [('a','b'),('b','a')]:
                for i,n in enumerate([128*1024,2*1024**2,8*1024**2]):
                    task=dict(key=f'pr5030-{os.getpid()}-{direction[0]}-{i}',bytes=n,offset=i+13)
                    inboxes[direction[0]].put(dict(task,op='put'));saved=receive('put')
                    inboxes[direction[1]].put(dict(task,op='get'));loaded=receive('get')
                    assert saved['sha256']==loaded['sha256']
            for inbox in inboxes.values():inbox.put(None)
            receive('closed');receive('closed')
            data['complete']=True;save()
        except BaseException:
            data['error']=traceback.format_exc();save();raise
        finally:
            for inbox in inboxes.values():inbox.put(None)
            for child in children:
                child.join(timeout=10)
                if child.is_alive():child.terminate();child.join(timeout=5)
                if child.is_alive():child.kill();child.join(timeout=5)
            data['child_exitcodes']=[child.exitcode for child in children];save()
            master.terminate()
            try:master.wait(timeout=10)
            except subprocess.TimeoutExpired:master.kill();master.wait(timeout=5)

if __name__=='__main__':main()

```
