# sitecustomize_native.py

原始验证文件快照；源码 SHA256：`80597e482ed5d8d0791347e9bd96190b806a2c59a263b29214b08b49ac111fcd`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Enforce no dlinfer in every native Python process, including engine workers."""
import importlib.abc
import importlib.machinery
import json
import os
import sys

if importlib.machinery.PathFinder.find_spec('dlinfer') is not None:
    os.write(2, b'FATAL: dlinfer discoverable in native environment\n')
    os._exit(96)

class NoDlinfer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'dlinfer':
            root = os.environ.get('VALIDATION_ROOT', '/root/workspace/a3-validation-20261112')
            with open(root + '/logs/forbidden-dlinfer-imports.jsonl', 'a') as f:
                f.write(json.dumps({'pid':os.getpid(), 'module':fullname})+'\n')
            raise ImportError('dlinfer import forbidden during native acceptance: '+fullname)

sys.meta_path.insert(0, NoDlinfer())

```
