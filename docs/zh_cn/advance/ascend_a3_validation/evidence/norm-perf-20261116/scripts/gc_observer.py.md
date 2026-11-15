# gc_observer.py

```python
"""Small callback recorder; no GC policy change unless benchmark requests it."""
import gc
import os
import time


class GCObserver:
    def __init__(self):
        self.events = []
        self.starts = {}
        gc.callbacks.append(self.callback)

    def callback(self, phase, info):
        now = time.monotonic()
        gen = info['generation']
        if phase == 'start':
            self.starts[gen] = now
        else:
            began = self.starts.pop(gen, now)
            if now-began > .005 or gen == 2:
                self.events.append(dict(began=began, ended=now, generation=gen,
                                        duration_ms=(now-began)*1000,
                                        collected=info['collected'], uncollectable=info['uncollectable']))

    def take(self):
        events, self.events = self.events, []
        return dict(pid=os.getpid(), enabled=gc.isenabled(), threshold=gc.get_threshold(),
                    counts=gc.get_count(), stats=gc.get_stats(), events=events)
```
