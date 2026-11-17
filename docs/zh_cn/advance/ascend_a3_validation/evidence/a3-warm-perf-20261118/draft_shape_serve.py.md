```python
"""Same API workload, with direct observation of both target and draft entries."""
from lmdeploy.pytorch.engine.executor import ray_executor
from warm_observer import install_driver
from draft_shape_observer import DraftShapeWorker

ray_executor.RayWorkerWrapper=DraftShapeWorker
install_driver()

if __name__=='__main__':
    from lmdeploy.cli import run
    run()
```
