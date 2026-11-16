import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import triton
import torch
import torch_npu
from triton.backends.ascend.utils import _get_npucompiler_path, _get_bisheng_path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), a.output
assert importlib.machinery.PathFinder.find_spec('dlinfer') is None
assert not any(x == 'dlinfer' or x.startswith('dlinfer.') for x in sys.modules)
selected, env = _get_npucompiler_path()
bins = [Path(selected), Path(selected).with_name('bishengir-opt'), Path(selected).with_name('hivmc'), Path(_get_bisheng_path())]
versions = {}
for path in bins:
    resolved = path.resolve()
    if not path.exists():
        versions[str(path)] = {'exists': False}
        continue
    version = subprocess.run([str(path), '--version'], capture_output=True, text=True, env=env)
    versions[str(path)] = dict(resolved=str(resolved), sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
                               version=version.stdout+version.stderr, returncode=version.returncode)
data = dict(python=sys.version, executable=sys.executable, triton_version=triton.__version__,
            triton_path=triton.__file__, torch=torch.__version__, torch_npu=torch_npu.__version__,
            distributions={x.metadata['Name']:x.version for x in importlib.metadata.distributions()
                           if x.metadata['Name'].lower().replace('_','-') in ('triton','triton-ascend','torch','torch-npu','ray','numpy')},
            selected_compiler=selected, compiler_subprocess_path=env['PATH'], binaries=versions,
            dlinfer_installed=False, dlinfer_imported=False, cwd=os.getcwd(),
            environment={k:os.environ.get(k) for k in ('PATH','LD_LIBRARY_PATH','PYTHONPATH','TRITON_CACHE_DIR',
                         'BISHENG_INSTALL_PATH','ASCEND_HOME_PATH','VIRTUAL_ENV')})
a.output.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
