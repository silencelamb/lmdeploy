# 保存输出的按位复核脚本

原始 trace 位于报告所列 A3 证据目录；该检查不会重新运行模型。
将下面代码保存为 `verify_recorded_output_bits.py`，运行：

```bash
python verify_recorded_output_bits.py /root/workspace/a3-native-pr-20261115/results/recorded-output-bitwise-audit.json
```

```python
"""Recheck saved outputs referenced by recorded-output-bitwise-audit.json."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

report = json.loads(Path(sys.argv[1]).read_text())
for case in report['results']:
    data = []
    for source in case['sources']:
        path = Path(source['path'])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256'], path
        data.append(json.loads(raw))
    old, new = data
    assert old['complete'] and new['complete']
    assert old['engine_config'] == new['engine_config']
    for key in ('model', 'mtp', 'new_tokens', 'suite', 'length',
                'no_thinking', 'respect_eos', 'skip_input_scoring'):
        assert old['options'].get(key) == new['options'].get(key), key
    assert len(old['events']) == len(new['events'])
    responses = tokens = rows = values = 0
    for left, right in zip(old['events'], new['events']):
        assert left['label'] == right['label']
        assert left.get('prompts') == right.get('prompts')
        assert len(left['responses']) == len(right['responses'])
        if left['label'].startswith('bench-'):
            continue
        for x, y in zip(left['responses'], right['responses']):
            responses += 1
            assert x['token_ids'] == y['token_ids']
            tokens += len(x['token_ids'])
            xs, ys = x.get('logprobs') or [], y.get('logprobs') or []
            assert len(xs) == len(ys)
            for a, b in zip(xs, ys):
                rows += 1
                assert a.keys() == b.keys()
                for token in a:
                    assert math.isfinite(a[token]) and math.isfinite(b[token])
                    assert struct.pack('>d', a[token]) == struct.pack('>d', b[token])
                    values += 1
    assert values > 0
    assert (responses, tokens, rows, values) == (
        case['responses'], case['generated_tokens'], case['score_rows'],
        case['score_values'])
    print(case['source_comparison'], 'PASS', responses, tokens, rows, values)
print('All', len(report['results']), 'saved-output cases passed.')
```
