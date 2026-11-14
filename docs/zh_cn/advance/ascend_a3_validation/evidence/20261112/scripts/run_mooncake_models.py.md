# run_mooncake_models.py

原始验证文件快照；源码 SHA256：`5d9282d59c0a2bcb0edaa74b37954c14f8aac9619818556dd292ff5b0c405e5a`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Attempt both deployment models with an owned, version-pinned Ascend Store."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+1800
while True:
    p=ROOT/'results/agents-http-progress.json'
    if p.exists() and len(json.loads(p.read_text()))==2:break
    if time.monotonic()>deadline:raise TimeoutError('Agents HTTP pair did not finish')
    time.sleep(2)
progress=[]
for model,mtp in [('Sage-A1.5',3),('Agents-A1',0)]:
    for env in ['baseline-mooncake','native-mooncake']:
        label=env+'-'+model.lower().replace('.','-')+'-fabric-model'
        binary=ROOT/'envs'/env/'lib/python3.12/site-packages/mooncake/mooncake_master'
        with socket.socket() as sock:sock.bind(('127.0.0.1',55051))
        with (ROOT/'logs'/f'{label}-master.log').open('w') as log:
            master=subprocess.Popen([str(binary),'-rpc_port=55051','-metrics_port=55052',
                '-rpc_thread_num=4','-default_kv_lease_ttl=30000','-eviction_high_watermark_ratio=0.95',
                '-eviction_ratio=0.1','-logtostderr'],stdout=log,stderr=subprocess.STDOUT)
            try:
                for _ in range(100):
                    assert master.poll() is None,'master exited'
                    try:
                        with socket.create_connection(('127.0.0.1',55051),timeout=.2):break
                    except OSError:time.sleep(.1)
                else:raise TimeoutError('master did not listen')
                cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',label,
                    '--devices','0,1','--timeout','300','--','env','ASCEND_ENABLE_USE_FABRIC_MEM=1',
                    'python',str(ROOT/'scripts/pipeline_case.py'),'--model','/afs-weights/'+model,
                    '--mtp',str(mtp),'--prefix','--scores','--suite','smoke','--no-thinking','--respect-eos',
                    '--kv-config',str(ROOT/'configs/mooncake-a3.json'),
                    '--output',str(ROOT/'results'/f'{label}.json')]
                print('START',label,flush=True)
                rc=subprocess.run(cmd).returncode
                progress.append(dict(label=label,returncode=rc))
                (ROOT/'results/mooncake-model-progress.json').write_text(json.dumps(progress,indent=2))
                print('DONE',label,rc,flush=True)
            finally:
                master.terminate()
                try:master.wait(timeout=10)
                except subprocess.TimeoutExpired:master.kill();master.wait()

```
