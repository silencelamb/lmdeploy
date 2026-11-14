# run.py

原始验证文件快照；源码 SHA256：`0674c3044acb4b49993a0a9a013b1a3d960c72ebd8065f2f377785a5561cca41`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Run a fully logged command in an isolated validation environment."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--env', choices=['baseline', 'native', 'standalone', 'baseline-mooncake', 'native-mooncake'], required=True)
p.add_argument('--label', required=True)
p.add_argument('--devices', default='0,1')
p.add_argument('--timeout', type=int, default=1800)
p.add_argument('--global-seed', type=int)
p.add_argument('--cwd', default=str(ROOT))
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[0] == '--' else a.command
env = os.environ.copy()
env.update(PATH=f'{ROOT}/envs/{a.env}/bin:' + env['PATH'],
           VIRTUAL_ENV=f'{ROOT}/envs/{a.env}', PYTHONNOUSERSITE='1',
           PYTHONPATH='/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe',
           ASCEND_RT_VISIBLE_DEVICES=a.devices, OMP_PROC_BIND='false', OMP_NUM_THREADS='1',
           TASK_QUEUE_ENABLE='1', HCCL_CONNECT_TIMEOUT='180',
           PYTORCH_NPU_ALLOC_CONF='expandable_segments:True', HCCL_OP_EXPANSION_MODE='AIV',
           HCCL_BUFFSIZE='512', RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE='1',
           LMDEPLOY_TARGET_DEVICE='ascend', DEVICE='ascend', DLINFER_BUILD_DICP='OFF',
           VALIDATION_ENV=a.env, VALIDATION_ROOT=str(ROOT),
           TRITON_CACHE_DIR=f'{ROOT}/artifacts/triton-cache-{a.env}',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
           PYTHONUNBUFFERED='1')
if a.global_seed is not None:
    env['VALIDATION_GLOBAL_SEED']=str(a.global_seed)
    env['PYTHONHASHSEED']=str(a.global_seed)
record = dict(label=a.label, env=a.env, command=cmd, cwd=a.cwd,
              environment={k:env[k] for k in env if k not in os.environ or env[k]!=os.environ[k]},
              start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
snapshot = ROOT / 'artifacts' / 'harness-snapshots' / a.label
snapshot.mkdir(parents=True,exist_ok=False)
record['harness_sha256']={}
for source in (ROOT/'scripts').glob('*.py'):
    shutil.copy2(source,snapshot/source.name)
    record['harness_sha256'][source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
projects={'baseline':['lmdeploy-baseline','dlinfer-baseline'],
          'native':['lmdeploy-native'],'standalone':['dlinfer-clean'],
          'baseline-mooncake':['lmdeploy-baseline-mooncake','dlinfer-baseline'],
          'native-mooncake':['lmdeploy-native-mooncake']}[a.env]
record['source_revisions']={}
for project in projects:
    path=ROOT/'sources'/project
    revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    patch=subprocess.check_output(['git','-C',str(path),'diff','--binary'])
    (snapshot/(project+'.patch')).write_bytes(patch)
    record['source_revisions'][project]={'head':revision,'diff_sha256':hashlib.sha256(patch).hexdigest()}
dest = ROOT / 'logs' / a.label
command_record=dest.parent/(dest.name+'.command.json')
assert not command_record.exists(), f'Duplicate label {a.label}'
command_record.write_text(json.dumps(record, indent=2))
start = time.monotonic()
with (dest.parent/(dest.name+'.log')).open('w') as log:
    child = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    record['pid'] = child.pid
    (dest.parent/(dest.name+'.pid')).write_text(str(child.pid))
    try:
        rc = child.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        record['timeout'] = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        rc = 124
record.update(returncode=rc, elapsed_seconds=time.monotonic()-start,
              end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
command_record.write_text(json.dumps(record, indent=2))
print(json.dumps({k:record[k] for k in ['label','returncode','elapsed_seconds']}, indent=2))
raise SystemExit(rc)

```
