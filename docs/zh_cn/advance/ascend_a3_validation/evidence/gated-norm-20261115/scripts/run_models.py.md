# run_models.py

```python
"""Sequential matched model runs; preserve failures and continue the matrix."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
repos = dict(before=str(ROOT/'lmdeploy-before'), after='/root/workspace/a3-native-pr-20261115/lmdeploy')
cases = [
    ('dense', 'Qwen3.5-0.8B-do01', 1, None, False),
    ('sage', 'Sage-A1.5', 2, None, True),
    ('agents', 'Agents-A1', 2, None, True),
    ('decision', 'Qwen3.5-0.8B-do01', 1, 'intern-decision', False),
    ('s2preview', 'Intern-S2-Preview', 2, None, False),
    ('s2mobius', 'Intern-S2-Mobius', 2, None, False),
    ('qwen3next', 'Qwen3-Next-80B-A3B-Instruct', 4, None, False),
]
records=[]
for name, checkpoint, tp, task, bench in cases:
    for side, repo in repos.items():
        label=f'{name}-{side}'
        command=[sys.executable, str(ROOT/'scripts/run.py'), '--label', label, '--repo', repo,
                 '--devices', ','.join(map(str,range(tp))), '--timeout', '900', '--',
                 str(ROOT/'scripts/pipeline_case.py'), '--model', f'/afs-weights/{checkpoint}',
                 '--tp', str(tp), '--scores', '--prefill', '256', '--session', '8192', '--batch', '8',
                 '--cache-fraction', '0.5', '--output', str(ROOT/'results'/f'{label}.json')]
        if task:
            command += ['--task', task]
        if bench:
            command += ['--suite', 'full', '--bench-concurrency', '1', '4', '--repeats', '3']
        record_path = ROOT/'logs'/f'{label}.json'
        if record_path.exists():
            record=json.loads(record_path.read_text())
            rc=record['returncode']
            print(f'REUSE {label} rc={rc}', flush=True)
        else:
            print(f'START {label}', flush=True)
            with (ROOT/'logs'/f'{label}-runner.log').open('w') as out:
                rc=subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode
        records.append(dict(label=label, returncode=rc, command=command))
        (ROOT/'results/model-runs.json').write_text(json.dumps(records,indent=2)+'\n')
        print(f'FINISH {label} rc={rc}', flush=True)
```
