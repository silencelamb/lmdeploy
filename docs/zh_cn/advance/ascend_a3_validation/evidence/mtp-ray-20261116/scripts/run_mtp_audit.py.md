# run_mtp_audit.py

```python
"""Use the existing trace workload with validation-only worker instrumentation."""
import os
import sys

if __name__=='__main__':
    assert os.environ.get('VALIDATION_AUDIT_LABEL')
    from lmdeploy.pytorch.engine.executor import ray_executor
    from mtp_audit import MTPAuditWorker
    ray_executor.RayWorkerWrapper=MTPAuditWorker
    from pipeline_case import main
    main()
```
