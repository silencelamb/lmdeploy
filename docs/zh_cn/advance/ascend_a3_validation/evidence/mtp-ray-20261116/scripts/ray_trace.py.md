# ray_trace.py

```python
"""Validation-only RPC timeline; preserve production execution order and awaits.

The input marker is removed before reaching the model. The otherwise unused
submission acknowledgement returns timing metadata. No tensor is copied to CPU.
All clocks are host CLOCK_MONOTONIC, so this probe is for one-node comparisons.
"""
import asyncio
import itertools
import json
import os
from pathlib import Path
import sys
import threading
import time

from lmdeploy.pytorch.engine.executor.ray_executor import RayWorkerWrapper

MARKER = '_validation_ray_trace'
ACK = '_validation_ray_ack'
_fd = None
_beats = {}


def record(kind, **data):
    global _fd
    if _fd is None:
        root = Path(os.environ['VALIDATION_ROOT']) / 'results'
        label = os.environ['VALIDATION_RAY_LABEL']
        _fd = os.open(root / f'ray-{label}-{os.getpid()}.jsonl',
                      os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    data.update(kind=kind, t_ns=time.monotonic_ns(), pid=os.getpid(),
                tid=threading.get_native_id())
    os.write(_fd, (json.dumps(data, separators=(',', ':')) + '\n').encode())


def heartbeat(role):
    loop = asyncio.get_running_loop()
    if id(loop) in _beats:
        return

    async def run():
        while True:
            expected = time.monotonic_ns() + 10_000_000
            await asyncio.sleep(0.01)
            now = time.monotonic_ns()
            if now - expected > 50_000_000:
                record('heartbeat_late', role=role, expected_ns=expected,
                       actual_ns=now, late_ms=(now-expected)/1e6)

    _beats[id(loop)] = loop.create_task(run())
    record('heartbeat_start', role=role, node=os.uname().nodename)


class TraceWorker(RayWorkerWrapper):
    async def forward_async(self, inputs):
        entered = time.monotonic_ns()
        meta = inputs.pop(MARKER)
        heartbeat(f'worker-{self.rank}')
        record('worker_enter', rank=self.rank, entered_ns=entered, **meta)
        await super().forward_async(inputs)
        ended = time.monotonic_ns()
        ack = dict(**{ACK: True}, **meta, rank=self.rank, worker_pid=os.getpid(),
                   worker_enter_ns=entered, worker_exit_ns=ended)
        record('worker_exit', **ack)
        return ack


def setup_driver():
    import ray
    from ray._private.serialization import SerializationContext
    from lmdeploy.pytorch.engine.executor import ray_executor

    ray_executor.RayWorkerWrapper = TraceWorker
    original_forward = ray_executor.RayExecutor.forward_async
    original_put = ray.put
    original_gather = asyncio.gather
    original_deserialize = SerializationContext.deserialize_objects
    counter = itertools.count(1)

    async def traced_forward(self, inputs):
        heartbeat('driver')
        item = next(counter)
        inputs = dict(inputs)
        inputs[MARKER] = dict(item=item, driver_pid=os.getpid())
        before = time.monotonic_ns()
        record('driver_forward_enter', item=item, entered_ns=before)
        try:
            return await original_forward(self, inputs)
        finally:
            record('driver_forward_exit', item=item, entered_ns=before,
                   duration_ms=(time.monotonic_ns()-before)/1e6)

    def traced_put(value, *args, **kwargs):
        if isinstance(value, dict) and MARKER in value:
            meta = value[MARKER]
            meta['dispatch_ns'] = time.monotonic_ns()
            record('driver_put_enter', **meta)
            result = original_put(value, *args, **kwargs)
            record('driver_put_exit', **meta)
            return result
        return original_put(value, *args, **kwargs)

    def traced_gather(*args, **kwargs):
        caller = sys._getframe(1)
        selected = (caller.f_code.co_name == 'forward_async'
                    and caller.f_code.co_filename.endswith('/ray_executor.py'))
        began = time.monotonic_ns()
        cpu_began = time.thread_time_ns()
        future = original_gather(*args, **kwargs)
        if selected:
            def completed(result):
                done = time.monotonic_ns()
                if result.cancelled():
                    record('driver_gather_cancelled', began_ns=began)
                    return
                error = result.exception()
                if error is not None:
                    record('driver_gather_error', error=repr(error))
                    return
                record('driver_gather_done', began_ns=began, done_ns=done,
                       wait_ms=(done-began)/1e6,
                       thread_cpu_ms=(time.thread_time_ns()-cpu_began)/1e6,
                       acknowledgements=result.result())
            future.add_done_callback(completed)
        return future

    def traced_deserialize(self, *args, **kwargs):
        began = time.monotonic_ns()
        results = original_deserialize(self, *args, **kwargs)
        done = time.monotonic_ns()
        for value in results:
            if isinstance(value, dict) and value.get(ACK):
                record('driver_ack_deserialized', deserialize_enter_ns=began,
                       deserialize_exit_ns=done, **value)
        return results

    ray_executor.RayExecutor.forward_async = traced_forward
    ray.put = traced_put
    asyncio.gather = traced_gather
    SerializationContext.deserialize_objects = traced_deserialize
    record('driver_setup', ray_version=ray.__version__, node=os.uname().nodename,
           cpu_arch=os.uname().machine)
```
