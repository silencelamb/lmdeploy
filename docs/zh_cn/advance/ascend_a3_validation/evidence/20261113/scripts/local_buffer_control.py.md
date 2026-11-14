# local_buffer_control.py

原始验证文件快照；源码 SHA256：`7fc1142c37ea10e5848cbb5b431344d6cb48ce29a0bc10bdf27e5e3bd2ff1f3e`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""One-variable control: same aligned segment, different effective local buffer."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from fabric_probe import master_argv

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-mib',required=True,type=int)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,default=55063)
    a=parser.parse_args()
    import torch
    import torch_npu
    import mooncake
    from mooncake.store import MooncakeDistributedStore
    from lmdeploy.pytorch.kv_connector.mooncake.store.worker import _get_local_hostname
    torch.npu.set_device(0);torch.ones(1,device='npu').cpu()
    for port in [a.port,a.port+1]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    data=dict(local_buffer_bytes=a.local_mib*1024**2,global_segment_bytes=1024**3,
        version=importlib.metadata.version('mooncake-transfer-engine-npu'),
        fabric=os.environ.get('ASCEND_ENABLE_USE_FABRIC_MEM'),setup_success=False)
    store=None
    with (ROOT/'logs'/f'{a.output.stem}-master.log').open('w') as log:
        master=subprocess.Popen(master_argv(Path(mooncake.__file__).parent/'mooncake_master',a.port),stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                assert master.poll() is None
                try:
                    with socket.create_connection(('127.0.0.1',a.port),timeout=.2):break
                except OSError:time.sleep(.1)
            else:raise TimeoutError('master not ready')
            store=MooncakeDistributedStore()
            ret=store.setup(local_hostname=_get_local_hostname(),metadata_server='P2PHANDSHAKE',
                global_segment_size=data['global_segment_bytes'],local_buffer_size=data['local_buffer_bytes'],
                protocol='ascend',rdma_devices='',master_server_addr=f'127.0.0.1:{a.port}')
            data['returncode']=ret;data['setup_success']=ret==0
            assert ret==0,ret
        except BaseException:
            data['error']=traceback.format_exc();raise
        finally:
            if data['setup_success']:data['close_returncode']=store.close()
            a.output.write_text(json.dumps(data,indent=2))
            master.terminate()
            try:master.wait(timeout=10)
            except subprocess.TimeoutExpired:master.kill();master.wait(timeout=5)

if __name__=='__main__':main()

```
