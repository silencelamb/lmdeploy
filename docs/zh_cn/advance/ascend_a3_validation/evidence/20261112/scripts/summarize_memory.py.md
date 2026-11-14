# summarize_memory.py

原始验证文件快照；源码 SHA256：`cbfd0c08ab901356baeea9181d2b419a8b333bc2bf4354221c2fd837f7bdbf1b`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
result={}
for env in ['baseline','native']:
    path=ROOT/'logs'/f'{env}-sage-http-recheck-npu.jsonl'
    maxima={};samples=0
    for line in path.read_text().splitlines():
        data=json.loads(line);samples+=1
        for row in data['stdout'].splitlines():
            match=re.search(r'^\|\s+(\d+)\s+(\d+)\s+\|\s+([0-9A-Fa-f:.]+)\s+\|.*?\s(\d+)\s*/\s*65536\s*\|$',row)
            if match:
                physical=match.group(2)
                maxima[physical]=max(maxima.get(physical,0),int(match.group(4)))
    result[env]=dict(samples=samples,sampled_hbm_peak_mb_by_physical_id=maxima,
        note='Total device HBM sampled roughly every 4 seconds, including resident driver memory. Not allocator maximum.')
(ROOT/'results/npu-memory-samples.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

```
