# run_final_cases.py

原始验证文件快照；源码 SHA256：`9303acaa0cac6f1da57d240f66adc8a5a28edc4a5717ee96564a38c94ff6e675`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
results=[]
cases=[('sage-http-fresh-connections','http_case.py',['--repeats','5']),
       ('sage-mtp3-prefix-quality','pipeline_case.py',['--mtp','3','--prefix','--scores','--suite','extra','--skip-input-scoring','--length','8192'])]
for label,script,args in cases:
    for env in ['baseline','native']:
        name=env+'-'+label
        cmd=[sys.executable,str(ROOT/'scripts/run.py'),'--env',env,'--label',name,'--timeout','1200','--','python',str(ROOT/'scripts'/script),*args,'--output',str(ROOT/'results'/f'{name}.json')]
        print('START',name,flush=True)
        rc=subprocess.run(cmd).returncode
        results.append(dict(name=name,returncode=rc))
        (ROOT/'results/final-cases-progress.json').write_text(json.dumps(results,indent=2))
        print('DONE',name,rc,flush=True)

```
