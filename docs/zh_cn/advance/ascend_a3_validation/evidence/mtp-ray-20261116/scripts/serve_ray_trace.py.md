# serve_ray_trace.py

```python
"""Start the ordinary LMDeploy CLI after installing validation-only probes."""
from ray_trace import setup_driver

setup_driver()

if __name__ == '__main__':
    from lmdeploy.cli import run
    run()
```
