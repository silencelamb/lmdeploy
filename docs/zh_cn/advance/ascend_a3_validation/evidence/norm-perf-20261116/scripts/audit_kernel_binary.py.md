# audit_kernel_binary.py

```python
"""Identify actual old/new compiled Agents gated-norm cache entries."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
records = []
binaries = {}
dest = ROOT/'inventory/norm-kernel'
dest.mkdir(parents=True, exist_ok=True)
for meta in sorted((ROOT/'artifacts/triton-cache').rglob('_rms_norm_fwd_kernel.json')):
    ir = meta.with_suffix('.ttir').read_text()
    side = 'old' if '/lmdeploy-before/' in ir else 'new'
    assert side not in binaries, 'Expected only the two actual BF16/128 specializations'
    binary = meta.with_suffix('.npubin').read_bytes()
    binaries[side] = binary
    record = dict(side=side, cache_path=str(meta.parent),
                  npubin_bytes=len(binary), npubin_sha256=hashlib.sha256(binary).hexdigest(),
                  metadata=json.loads(meta.read_text()),
                  source='342db85ecff3c0dc864a4bf8613c589c270f4141' if side=='old' else '5f1b1b3c12b1c198813f65275f778c7ad292fb02')
    for ext in ('ttir','ttadapter'):
        source = meta.with_suffix('.'+ext)
        (dest/f'{side}.{ext}.md').write_text(f'# {side} {ext}\n\n```text\n'+source.read_text()+'\n```\n')
        record[ext+'_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    records.append(record)
assert set(binaries)=={'old','new'}
result=dict(byte_identical=binaries['old']==binaries['new'],specializations=records,
            scope='Actual Agents BF16 width-128 SiLU, affine, no bias, norm-before-gate; rows are runtime arguments. Not sigmoid or other dtypes.')
(ROOT/'results/kernel-binary-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['byte_identical'], [(r['side'],r['npubin_bytes'],r['npubin_sha256']) for r in records])
```
