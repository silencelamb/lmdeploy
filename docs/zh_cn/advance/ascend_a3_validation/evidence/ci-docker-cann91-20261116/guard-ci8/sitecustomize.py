import importlib.abc
import importlib.machinery
import json
import os
import sys

if importlib.machinery.PathFinder.find_spec('dlinfer') is not None:
    os.write(2, b'FATAL: dlinfer discoverable in review environment\n')
    os._exit(96)

class ReviewImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'dlinfer':
            raise AssertionError('Forbidden dlinfer import: ' + fullname)

sys.meta_path.insert(0, ReviewImportGuard())

# Checkpoint tokenizer dependency must load before torch_npu in this image.
if os.environ.get("VALIDATION_PRELOAD_RDKIT") == "1":
    from rdkit import Chem  # noqa: F401

class BlockExternalAscendKernels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'triton_ascend_kernels':
            raise AssertionError('Native prefill must use the bundled source: ' + fullname)
sys.meta_path.insert(0, BlockExternalAscendKernels())
