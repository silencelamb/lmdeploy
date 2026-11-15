# check_checkpoint_scales.py

```python
"""Check every gated norm scale in representative real checkpoints on Ascend."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

from lmdeploy.pytorch.backends.ascend.runtime import initialize
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.nn import RMSNormGated

ROOT = Path(__file__).resolve().parents[1]
initialize()
old_path = ROOT / 'lmdeploy-before/lmdeploy/pytorch/kernels/ascend/triton_ops/rms_norm_gated.py'
spec = importlib.util.spec_from_file_location('frozen_checkpoint_norm', old_path)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
models = ['Sage-A1.5', 'Agents-A1', 'Qwen3-Next-80B-A3B-Instruct', 'Qwen3.5-0.8B-do01',
          'Intern-S2-Preview', 'Intern-S2-Mobius', 'GLM-5.3-Flash']
report = dict(baseline_commit='342db85ecff3c0dc864a4bf8613c589c270f4141', models=[], all_passed=False)
torch.manual_seed(137)

def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

with torch.inference_mode(), get_device_manager().context(DeviceContext('ascend')):
    for model in models:
        root = Path('/afs-weights') / model
        activation = 'sigmoid' if model == 'GLM-5.3-Flash' else 'silu'
        selected = defaultdict(list)
        def is_gate(name):
            return name.endswith('.linear_attn.norm.weight') or name.endswith('.o_norm.weight')
        index = root / 'model.safetensors.index.json'
        if index.exists():
            mapping = json.loads(index.read_text())['weight_map']
            for name, shard in mapping.items():
                if is_gate(name):
                    selected[shard].append(name)
        else:
            for shard in sorted(root.glob('*.safetensors')):
                with safe_open(shard, framework='pt', device='cpu') as f:
                    selected[shard.name] = [name for name in f.keys() if is_gate(name)]
        item = dict(model=model, activation=activation,
                    config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(), cases=[])
        for shard, names in sorted(selected.items()):
            if not names:
                continue
            with safe_open(root/shard, framework='pt', device='cpu') as f:
                for name in sorted(names):
                    scale = f.get_tensor(name)
                    assert scale.ndim == 1 and scale.dtype in (torch.bfloat16, torch.float16, torch.float32), (name, scale.shape, scale.dtype)
                    width = scale.numel()
                    layer = RMSNormGated(width, dtype=scale.dtype, device='npu', activation=activation)
                    layer.load_state_dict({'weight': scale}, strict=True)
                    assert digest(layer.weight) == digest(scale)
                    x, z = (torch.randn(17, width, device='npu', dtype=scale.dtype) for _ in range(2))
                    actual = layer(x, z)
                    legacy = old.rmsnorm_fn(x, layer.weight, None, z=z, norm_before_gate=True)
                    if activation == 'silu':
                        assert digest(actual) == digest(legacy), name
                        verification = dict(bitwise_equal=True)
                    else:
                        reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                                     * layer.weight.float() * torch.sigmoid(z.float())).to(x.dtype)
                        tolerance = {torch.bfloat16: .02, torch.float16: .003, torch.float32: 1e-5}[scale.dtype]
                        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
                        verification = dict(reference_passed=True,
                                            max_abs_error=(actual.float()-reference.float()).abs().max().item(),
                                            differs_from_legacy_ignored_activation=digest(actual) != digest(legacy))
                    item['cases'].append(dict(name=name, shard=shard, dtype=str(scale.dtype), width=width,
                                              weight_sha256=digest(scale), output_sha256=digest(actual), **verification))
        assert item['cases'], model
        report['models'].append(item)
        print(json.dumps(dict(model=model, activation=activation, layers=len(item['cases']), passed=True)), flush=True)
report['all_passed'] = True
(ROOT/'results/checkpoint-scales.json').write_text(json.dumps(report, indent=2)+'\n')
```
