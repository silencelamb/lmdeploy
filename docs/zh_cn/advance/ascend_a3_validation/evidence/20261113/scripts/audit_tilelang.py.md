# audit_tilelang.py

原始验证文件快照；源码 SHA256：`3c6bfeae9150fc4dfb61cc340563c53b3fb02096dd0e4bed08f78e0eb59ab691`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Verify installed pins, declared dependencies, and the existing FFI consumer."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
name=os.environ['VALIDATION_ENV']
selected=json.loads((ROOT/'inventory/tilelang-followup/selected-wheels.json').read_text())
versions={item['name']:md.version(item['name']) for item in selected}
assert all(versions[item['name']]==item['version'] for item in selected)
bad=[]
for item in selected+[dict(name='xgrammar')]:
    for raw in md.requires(item['name']) or []:
        requirement=Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
        try:version=md.version(requirement.name)
        except md.PackageNotFoundError:bad.append([item['name'],raw,'missing']);continue
        if requirement.specifier and version not in requirement.specifier:bad.append([item['name'],raw,version])
assert not bad,bad
import tilelang
import tvm_ffi
import xgrammar
grammar=xgrammar.Grammar.from_ebnf('root ::= "ok"')
assert grammar is not None
modules=[key for key in sys.modules if key.split('.')[0]=='dlinfer']
distributions=[d.metadata['Name'] for d in md.distributions() if 'dlinfer' in d.metadata['Name'].lower()]
if name=='native':assert not modules and not distributions
check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
old=json.loads((ROOT/'inventory'/f'{name}-environment.json').read_text())
added_errors=sorted(set(check.stdout.splitlines())-set(old['pip-check']['stdout'].splitlines()))
assert not added_errors,added_errors
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
result=dict(environment=name,versions=versions,tilelang_version=tilelang.__version__,
    tilelang_path=tilelang.__file__,tvm_ffi_path=tvm_ffi.__file__,xgrammar_grammar_parse=True,
    new_dependency_errors=bad,pip_check_returncode=check.returncode,pip_check_stdout=check.stdout,
    added_pip_check_errors=added_errors,dlinfer_modules=modules,dlinfer_distributions=distributions)
(ROOT/'inventory/tilelang-followup'/f'{name}-audit.json').write_text(json.dumps(result,indent=2))
(ROOT/'inventory/tilelang-followup'/f'{name}-freeze.txt').write_text(freeze)
print(json.dumps(result,indent=2))

```
