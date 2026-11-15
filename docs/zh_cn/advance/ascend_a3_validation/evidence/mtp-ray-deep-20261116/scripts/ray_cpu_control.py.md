# ray_cpu_control.py

```python
"""Portable Ray-only control: no model, Torch, NPU or CUDA execution.

Two async actors receive the same payload, like the LMDeploy TP submission RPCs.
Alternate ray.put/free and direct argument blocks. A fixed 12 ms driver sleep
stands in for waiting for model output; it is not a model or performance test.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import ray


@ray.remote(num_cpus=0)
class Echo:
    async def submit(self, value):
        entered = time.monotonic_ns()
        return dict(item=value['item'], pid=os.getpid(), entered_ns=entered,
                    dispatch_ns=value['dispatch_ns'],
                    data_size=value['data'].size,
                    torch_imported='torch' in sys.modules,
                    npu_imported='torch_npu' in sys.modules)


async def run(args, actors):
    records = []
    item = 0
    for block, mode in enumerate(['put', 'direct', 'direct', 'put']):
        previous = None
        input_ref = None

        async def finish_previous():
            nonlocal previous, input_ref
            if previous is None:
                return
            began = time.monotonic_ns()
            replies = await asyncio.gather(*previous)
            ended = time.monotonic_ns()
            records.append(dict(block=block, mode=mode, began_ns=began, done_ns=ended,
                                wait_ms=(ended-began)/1e6, replies=replies))
            if input_ref is not None:
                ray.internal.free(input_ref, local_only=False)
                input_ref = None

        for _ in range(args.iterations):
            await finish_previous()
            item += 1
            value = dict(item=item, dispatch_ns=time.monotonic_ns(),
                         data=np.arange(args.payload_bytes, dtype=np.uint8))
            if mode == 'put':
                input_ref = ray.put(value)
                previous = [actor.submit.remote(input_ref) for actor in actors]
            else:
                previous = [actor.submit.remote(value) for actor in actors]
            await asyncio.sleep(args.interval_ms/1000)
        await finish_previous()
        print('BLOCK', block, mode, args.iterations, flush=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=1500)
    p.add_argument('--interval-ms', type=float, default=12)
    p.add_argument('--payload-bytes', type=int, default=8192)
    args = p.parse_args()
    assert not args.output.exists(), args.output
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
    try:
        actors = [Echo.remote() for _ in range(2)]
        # Exclude actor startup before examining a submission-latency spike.
        ray.get([actor.submit.remote(dict(item=0, dispatch_ns=time.monotonic_ns(),
                                         data=np.zeros(1, dtype=np.uint8))) for actor in actors])
        records = asyncio.run(run(args, actors))
        data = dict(ray_version=ray.__version__, cpu_arch=platform.machine(),
                    python=sys.version, options=vars(args) | {'output': str(args.output)},
                    torch_imported='torch' in sys.modules, npu_imported='torch_npu' in sys.modules,
                    records=records, complete=True)
        args.output.write_text(json.dumps(data, indent=2)+'\n')
        for mode in ('put', 'direct'):
            rows = [r for r in records if r['mode']==mode]
            delivery = [(q['entered_ns']-q['dispatch_ns'])/1e6 for r in rows for q in r['replies']]
            print(mode, 'joins', len(rows), 'max_wait_ms', max(r['wait_ms'] for r in rows),
                  'max_delivery_ms', max(delivery), 'delivery_over_100ms', sum(x>100 for x in delivery))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
```
