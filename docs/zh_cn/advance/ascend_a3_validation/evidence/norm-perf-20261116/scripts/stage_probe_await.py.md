# stage_probe_await.py

```python
"""Validation-only timers around host dispatch, metadata and model boundaries."""
import functools
import inspect
import os
import threading
import time

from lmdeploy.pytorch.engine.executor.ray_executor import RayWorkerWrapper

RECORDS = []


def stamp():
    with open('/proc/thread-self/schedstat') as f:
        run, queued, switches = map(int, f.read().split())
    return time.monotonic(), time.thread_time(), run, queued, switches


def finish(name, before):
    after = stamp()
    elapsed = (after[0]-before[0])*1000
    if elapsed > 50:
        RECORDS.append(dict(name=name, began=before[0], ended=after[0], wall_ms=elapsed,
            thread_cpu_ms=(after[1]-before[1])*1000, runqueue_ms=(after[3]-before[3])/1e6,
            switches=after[4]-before[4], tid=threading.get_native_id()))


def wrap(owner, name, label=None):
    fn = getattr(owner, name)
    label = label or name
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def timed(*args, **kwargs):
            before = stamp()
            try:
                return await fn(*args, **kwargs)
            finally:
                finish(label, before)
    else:
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            before = stamp()
            try:
                return fn(*args, **kwargs)
            finally:
                finish(label, before)
    setattr(owner, name, timed)


def describe():
    return dict(pid=os.getpid(), threads=[dict(name=t.name, tid=t.native_id,
        affinity=sorted(os.sched_getaffinity(t.native_id))) for t in threading.enumerate() if t.native_id])


def take():
    global RECORDS
    records, RECORDS = RECORDS, []
    return dict(process=describe(), slow_stages=records)


def setup_driver():
    import ray
    from lmdeploy.pytorch.engine import inputs_maker
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor
    wrap(inputs_maker.InputsMakerAsync, '_make_forward_inputs', 'driver.make_inputs')
    wrap(RayExecutor, 'forward_async', 'driver.submit_inputs')
    wrap(RayExecutor, 'get_output_async', 'driver.await_outputs')
    wrap(ray, 'put', 'driver.ray_put')
    wrap(ray.internal, 'free', 'driver.ray_free')
    wrap(ray.actor.ActorMethod, '_remote', 'driver.actor_remote')
    import asyncio
    import sys
    original_gather = asyncio.gather
    def timed_gather(*args, **kwargs):
        caller = sys._getframe(1)
        selected = caller.f_code.co_name == 'forward_async' and caller.f_code.co_filename.endswith('/ray_executor.py')
        before = stamp() if selected else None
        future = original_gather(*args, **kwargs)
        if selected:
            future.add_done_callback(lambda result: finish('driver.previous_forward_rpc', before))
        return future
    asyncio.gather = timed_gather
    return describe()


class AwaitStageWorker(RayWorkerWrapper):
    def stage_setup(self):
        from lmdeploy.pytorch.backends.ascend import op_backend
        from lmdeploy.pytorch.backends.ascend.graph_runner import AscendGraphRunner
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, BatchedOutputs
        wrap(op_backend, '_get_cpu_attention_metadata', 'worker.cpu_attention_metadata')
        wrap(AscendGraphRunner, '__call__', 'worker.model_enqueue')
        wrap(BaseModelAgent, '_forward_impl', 'worker.forward_with_metadata')
        wrap(BaseModelAgent, 'async_sampling_logits', 'worker.sampling')
        wrap(BaseModelAgent, 'get_output_async', 'worker.await_outputs')
        wrap(BatchedOutputs, 'to_cpu', 'worker.output_to_cpu')
        return describe()

    def stage_collect(self):
        return take()
```
