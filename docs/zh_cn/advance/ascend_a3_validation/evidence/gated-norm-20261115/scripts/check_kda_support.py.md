# check_kda_support.py

```python
"""Record the pre-existing independent blocker to full GLM5-Next on Ascend."""
import json
from lmdeploy.pytorch.backends.ascend.op_backend import AscendOpsBackend
from lmdeploy.pytorch.backends.kda import KdaBuildSpec

try:
    impl = AscendOpsBackend.build_op(KdaBuildSpec())
except RuntimeError as exc:
    assert 'KdaBuildSpec is not supported by ascend backend' in str(exc)
    print(json.dumps(dict(supported=False, error_type=type(exc).__name__, message=str(exc))))
else:
    print(json.dumps(dict(supported=True, implementation=type(impl).__module__)))
```
