# run_gdn_regression_direct.py

```python
"""Invoke the same hardware regression without pytest changing repository roots."""
from pathlib import Path
import runpy
import torch
import torch_npu
from lmdeploy.pytorch.backends.ascend.runtime import initialize
import lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent as kernel

print('KERNEL_SOURCE',kernel.__file__,flush=True)
assert Path(kernel.__file__).is_relative_to(Path.cwd()), (kernel.__file__,Path.cwd())
initialize()
test=runpy.run_path('/root/workspace/a3-mtp-ray-investigation-20261116/lmdeploy-research/tests/pytorch/ascend/hardware/test_gdn_state_writeback.py')
test['test_circular_state_writeback'](4,True)
print('FIXED_INPUT_GRAPH_REGRESSION_PASSED',flush=True)
```
