"""Run the PR's actual regression file and record the circular launch policy."""
from pathlib import Path
import sys

import pytest
import torch
import torch_npu
import lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent as kernel

assert Path(kernel.__file__).is_relative_to(Path.cwd())
print('PINNED_KERNEL_SOURCE',kernel.__file__,flush=True)
original = kernel.fused_recurrent_gated_delta_rule_fwd_kernel
stages = set()

class RecordLaunch:
    def __getitem__(self,grid):
        call = original[grid]
        def run(*args,**kwargs):
            stages.add(kwargs['num_stages'])
            return call(*args,**kwargs)
        return run

kernel.fused_recurrent_gated_delta_rule_fwd_kernel = RecordLaunch()
rc = pytest.main(['-q','--run-ascend-hardware','--maxfail=1',
                 str(Path.cwd()/'tests/pytorch/ascend/hardware/test_gdn_state_writeback.py')])
print('OBSERVED_CIRCULAR_NUM_STAGES',sorted(stages),flush=True)
if rc == 0: assert stages == {1}
raise SystemExit(rc)
