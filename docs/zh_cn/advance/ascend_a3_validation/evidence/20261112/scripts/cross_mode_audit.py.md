# cross_mode_audit.py

原始验证文件快照；源码 SHA256：`31b6cf2389c9f33eca177efd25c10afe424dc8dbfd77dbdb21678743818b1839`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Keep the eager versus graph/chunk difference separate from migration parity."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
result={}
for env in ['baseline','native']:
    old=json.loads((ROOT/'results'/f'{env}-sage-eager-score-trace.json').read_text())
    new=json.loads((ROOT/'results'/f'{env}-sage-graph-chunk256-score-trace.json').read_text())
    total=exact=0
    mismatches=[]
    for e,f in zip(old['events'],new['events']):
        assert e['label']==f['label'] and e['prompts']==f['prompts']
        for i,(a,b) in enumerate(zip(e['responses'],f['responses'])):
            total+=1
            exact+=int(a['token_ids']==b['token_ids'])
            for j,(u,v) in enumerate(zip(a['token_ids'],b['token_ids'])):
                if u!=v:
                    mismatches.append(dict(event=e['label'],request=i,position=j,
                                           eager_token=u,graph_chunk256_token=v))
                    break
    result[env]=dict(total=total,exact=exact,first_mismatches=mismatches)
result['same_mismatches_on_both_branches']=result['baseline']==result['native']
result['scope']='Changes both eager/graph and prefill chunk size; not a single-variable graph diagnosis. Each same-mode migration pair passes separately.'
(ROOT/'results/cross-mode-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

```
