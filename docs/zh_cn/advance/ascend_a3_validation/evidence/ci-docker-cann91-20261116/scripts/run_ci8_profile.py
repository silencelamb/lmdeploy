"""Run isolated, logged gated-norm checks and model comparisons."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--cpu', action='store_true')
p.add_argument('--venv', default=str(ROOT / 'envs/upgrade5-full322p2'))
p.add_argument('--compiler-cache', default='ci8-new')
p.add_argument('--repo', default=str(ROOT / 'lmdeploy-ci8'))
p.add_argument('--devices', default='0')
p.add_argument('--timeout', type=int, default=900)
p.add_argument('args', nargs=argparse.REMAINDER)
a = p.parse_args()
venv = Path(a.venv)
env = os.environ.copy()
env.update(PYTHONPATH=f'{ROOT}/guard-ci8:{ROOT}/artifacts/ci8-pyprofile:{ROOT}/scripts:{a.repo}:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe',
           PATH=str(venv / 'bin') + ':' + env['PATH'], VIRTUAL_ENV=str(venv),
           PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
           VALIDATION_ENV='review', VALIDATION_ROOT=str(ROOT), ASCEND_RT_VISIBLE_DEVICES=a.devices,
           OMP_NUM_THREADS='1', OMP_PROC_BIND='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', TASK_QUEUE_ENABLE='1',
           HCCL_CONNECT_TIMEOUT='180', HCCL_OP_EXPANSION_MODE='AIV', HCCL_BUFFSIZE='512',
           PYTORCH_NPU_ALLOC_CONF='expandable_segments:True', LMDEPLOY_TARGET_DEVICE='ascend',
           TRITON_CACHE_DIR=str(ROOT / 'artifacts' / (a.compiler_cache + '-triton-cache')),
           TRITON_DUMP_DIR=str(ROOT / 'artifacts' / (a.label + '-triton-dump')))
os.sched_setaffinity(0, os.sched_getaffinity(0) - {18})
args = a.args[1:] if a.args and a.args[0] == '--' else a.args
if '--preload-rdkit' in args:
    env['VALIDATION_PRELOAD_RDKIT'] = '1'
command = [str(venv / 'bin/python'), *args]
record = dict(label=a.label, command=command, cwd=a.repo, cpu=a.cpu,
              cpu_affinity=sorted(os.sched_getaffinity(0)),
              start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.repo, text=True).strip(),
              environment={key: value for key, value in env.items() if os.environ.get(key) != value})
diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=a.repo)
(ROOT / 'artifacts' / f'{a.label}.patch').write_bytes(diff)
record['diff_sha256'] = hashlib.sha256(diff).hexdigest()
log = ROOT / 'logs' / f'{a.label}.log'
assert not log.exists(), log
record['diagnostic_environment'] = {k:v for k,v in env.items() if k.startswith(('RAY_','GRPC_','MTP_','VALIDATION_')) or k in ('PYTHONHASHSEED','LD_PRELOAD','LD_LIBRARY_PATH')}
snapshot = ROOT/'artifacts'/f'{a.label}-scripts'
snapshot.mkdir(exist_ok=False)
record['scripts'] = {}
for source in (ROOT/'scripts').glob('*.py'):
    raw=source.read_bytes()
    (snapshot/source.name).write_bytes(raw)
    record['scripts'][source.name]=hashlib.sha256(raw).hexdigest()
record['venv'] = str(venv)
record['compiler_cache'] = env['TRITON_CACHE_DIR']
start = time.monotonic()
with log.open('w') as output:
    child = subprocess.Popen(command, cwd=a.repo, env=env, stdout=output, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        rc = child.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        rc = 124
record.update(returncode=rc, elapsed_seconds=time.monotonic() - start)
(ROOT / 'logs' / f'{a.label}.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: record[k] for k in ('label', 'cwd', 'venv', 'returncode', 'elapsed_seconds')}, indent=2))
print('\n'.join(log.read_text().splitlines()[-28:]))
raise SystemExit(rc)
