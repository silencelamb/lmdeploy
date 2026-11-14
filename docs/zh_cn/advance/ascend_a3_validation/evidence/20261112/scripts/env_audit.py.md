# env_audit.py

原始验证文件快照；源码 SHA256：`3bb31744994609a9844611109abb9fc793db69203a9b038d7e8685804cef3dfd`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(os.environ['VALIDATION_ROOT'])
name=os.environ['VALIDATION_ENV']
dist={d.metadata['Name']:d.version for d in md.distributions()}
res={'name':name,'executable':sys.executable,'sys_path':sys.path,'packages':dist,
     'environment':{k:v for k,v in os.environ.items() if k.startswith(('ASCEND','HCCL','PYTHON','DLINFER','TRITON','OMP','TASK_QUEUE','LD_'))}}
for kind,args in [('pip-check',['check']),('freeze',['freeze','--all'])]:
    p=subprocess.run([sys.executable,'-m','pip',*args],text=True,capture_output=True)
    res[kind]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
import torch
import torch_npu
res.update(torch=torch.__version__,torch_npu=torch_npu.__version__,soc=torch.npu.get_soc_version(),device_count=torch.npu.device_count())
if name.startswith('native'):
    assert not any('dlinfer' in n.lower() for n in dist)
    from lmdeploy.pytorch.backends.ascend.runtime import initialize
    initialize()
    from lmdeploy.pytorch.backends import get_backend
    res['backend']=get_backend('ascend').__module__
    for n in ['causal_conv1d','rms_norm_gated','fla.chunk','fla.fused_recurrent','fla.sigmoid_gating','reject_sample']:
        __import__('lmdeploy.pytorch.kernels.ascend.triton_ops.'+n)
else:
    import dlinfer.ops
    res['dlinfer']=dlinfer.__file__
    if name=='standalone':
        import dlinfer.framework.lmdeploy_ext
        assert not any(n.split('.')[0]=='lmdeploy' for n in sys.modules)
        assert importlib.util.find_spec('lmdeploy') is None
res['lmdeploy_modules']=[n for n in sys.modules if n.split('.')[0]=='lmdeploy']
res['dlinfer_modules']=[n for n in sys.modules if n.split('.')[0]=='dlinfer']
(root/'inventory'/f'{name}-environment.json').write_text(json.dumps(res,indent=2))
print(json.dumps({k:res[k] for k in ['name','torch','torch_npu','soc','device_count','pip-check']},indent=2))

```
