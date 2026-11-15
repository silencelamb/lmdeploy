# run_gdn_pytest_pinned.py

```python
"""Pin the imported implementation before collecting tests from another checkout."""
from pathlib import Path
import sys
import torch
import torch_npu
import pytest
import lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent as kernel

print('PINNED_KERNEL_SOURCE',kernel.__file__,flush=True)
assert Path(kernel.__file__).is_relative_to(Path.cwd())
raise SystemExit(pytest.main(['-q','--run-ascend-hardware','--maxfail=1',
    '/root/workspace/a3-mtp-ray-investigation-20261116/lmdeploy-research/tests/pytorch/ascend/hardware/test_gdn_state_writeback.py',*sys.argv[1:]]))
```
