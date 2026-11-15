# compare_legacy_kernel.py

```python
"""Require bitwise agreement with the frozen pre-refactor Ascend kernel."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from lmdeploy.pytorch.backends.ascend.runtime import initialize
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.kernels.ascend.triton_ops.rms_norm_gated import rmsnorm_fn
from lmdeploy.pytorch.nn import RMSNormGated

ROOT = Path(__file__).resolve().parents[1]
initialize()
old_path = ROOT / 'lmdeploy-before/lmdeploy/pytorch/kernels/ascend/triton_ops/rms_norm_gated.py'
spec = importlib.util.spec_from_file_location('frozen_gated_norm', old_path)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
results = []


def same_bits(label, actual, expected):
    left = actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    right = expected.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    result = dict(label=label, shape=list(actual.shape), dtype=str(actual.dtype),
                  equal=left == right, sha256=hashlib.sha256(left).hexdigest())
    results.append(result)
    assert result['equal'], result


with torch.inference_mode():
    torch.manual_seed(137)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for width in (128, 192):
            for rows in (1, 17, 4097):
                x, z = (torch.randn(rows, width * 2, device='npu', dtype=dtype)[:, ::2] for _ in range(2))
                weight = torch.randn(width, device='npu', dtype=dtype)
                before = old.rmsnorm_fn(x, weight, None, z=z)
                after = rmsnorm_fn(x, weight, None, z=z)
                same_bits(f'kernel-{dtype}-{rows}x{width}', after, before)
        x, z = (torch.randn(17, 192, device='npu', dtype=dtype) for _ in range(2))
        weight, bias = (torch.randn(192, device='npu', dtype=dtype) for _ in range(2))
        for gate in (None, z):
            for order in (False, True):
                kwargs = dict(z=gate, group_size=64, norm_before_gate=order)
                same_bits(f'grouped-{dtype}-gate{gate is not None}-normfirst{order}',
                          rmsnorm_fn(x, weight, bias, **kwargs), old.rmsnorm_fn(x, weight, bias, **kwargs))
    for width in (128, 192):
        before = old.RMSNormGated(width, eps=1e-6, norm_before_gate=True, device='npu')
        with get_device_manager().context(DeviceContext('ascend')):
            after = RMSNormGated(width, eps=1e-6, device='npu', dtype=torch.bfloat16)
        before.weight.copy_(torch.randn_like(before.weight))
        after.load_state_dict(before.state_dict(), strict=True)
        x, z = (torch.randn(17, width, device='npu', dtype=torch.bfloat16) for _ in range(2))
        same_bits(f'public-module-{width}', after(x, z), before(x, z))

report = dict(baseline_commit='342db85ecff3c0dc864a4bf8613c589c270f4141',
              baseline_kernel_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
              cases=results, all_equal=all(item['equal'] for item in results))
(ROOT / 'results/legacy-kernel-bitwise.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'cases': len(results), 'all_equal': report['all_equal']}))
```
