# seed_startup.py

原始验证文件快照；源码 SHA256：`81b45b6fc2f484b00b86577a7bd7a55f9b86c71bb1aaee1e03641ce1188f4cae`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Optional diagnostic seed for every interpreter, including Ray workers."""
import os

if os.environ.get('VALIDATION_GLOBAL_SEED') is not None:
    import random
    import numpy
    import torch
    seed=int(os.environ['VALIDATION_GLOBAL_SEED'])
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

```
