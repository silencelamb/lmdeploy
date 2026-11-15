# serve_ray_trace-r0-failed.py

```python
"""Start the ordinary LMDeploy CLI after installing validation-only probes."""
from ray_trace import setup_driver

setup_driver()

from lmdeploy.cli import run

run()
```
