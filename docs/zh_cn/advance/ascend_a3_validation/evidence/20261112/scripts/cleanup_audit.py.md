# cleanup_audit.py

原始验证文件快照；源码 SHA256：`6f6f380ed0eb3b0c520eb86d0107d5bb48245036445afe6dab894d3b1b7d5516`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Read-only audit of the services and NPU processes used by this validation."""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ancestors = set()
pid = os.getpid()
while pid > 0 and pid not in ancestors:
    ancestors.add(pid)
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
    except (FileNotFoundError, StopIteration):
        break

processes = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    try:
        argv = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if any(marker in argv for marker in [str(ROOT), 'mooncake_master', 'lmdeploy serve', 'ray::']):
            processes.append(dict(pid=int(entry.name), argv=argv))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass

listeners = []
for table in ['/proc/net/tcp', '/proc/net/tcp6']:
    for line in Path(table).read_text().splitlines()[1:]:
        fields = line.split()
        port = int(fields[1].rsplit(':', 1)[1], 16)
        if fields[3] == '0A' and port in {25485, 55051, 55052}:
            listeners.append(dict(table=table, port=port, local=fields[1], inode=fields[9]))

npu = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True)
(ROOT / 'inventory/npu-smi-final.txt').write_text(npu.stdout + npu.stderr)
no_npu_processes = npu.returncode == 0 and npu.stdout.count('No running processes found in NPU') == 4
commands = list((ROOT / 'logs').glob('*.command.json'))
unfinished = [p.name for p in commands if 'returncode' not in json.loads(p.read_text())]
result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              matching_processes=processes, validation_port_listeners=listeners,
              npu_smi_returncode=npu.returncode, no_npu_processes=no_npu_processes,
              command_records=len(commands), unfinished_commands=unfinished,
              passed=not processes and not listeners and no_npu_processes and not unfinished)
(ROOT / 'inventory/cleanup-final.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result['passed']

```
