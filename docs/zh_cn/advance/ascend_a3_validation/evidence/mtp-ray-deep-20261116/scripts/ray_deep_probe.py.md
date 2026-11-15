# ray_deep_probe.py

```python
"""CPU-only Ray probe. Keeps submit/sleep/join/free ordering of the original.

Serialization marks are monotonic on this single host. No Torch/NPU import.
Environment controls (e.g. GRPC_POLL_STRATEGY) must be set before startup.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time

import numpy as np
import ray


def restore_packet(value, kind, serialize_ns, serializer_tid):
    start = time.monotonic_ns()
    packet = Packet(value, kind)
    packet.stamps = dict(serialize_ns=serialize_ns, serializer_tid=serializer_tid,
                         deserialize_ns=start, deserialize_tid=threading.get_native_id(),
                         deserialize_end_ns=time.monotonic_ns())
    return packet


class Packet:
    def __init__(self, value, kind):
        self.value, self.kind = value, kind
        self.stamps = {}

    def __reduce__(self):
        return restore_packet, (self.value, self.kind, time.monotonic_ns(), threading.get_native_id())


async def heartbeat(rows):
    while True:
        start = time.monotonic_ns()
        await asyncio.sleep(.01)
        end = time.monotonic_ns()
        # Save all >5ms lateness, not just >50ms.
        if end-start > 15_000_000:
            rows.append(dict(start_ns=start, end_ns=end, late_ms=(end-start)/1e6-10))


@ray.remote(num_cpus=0)
class Echo:
    def __init__(self):
        self.beats = []
        self.task = None

    async def submit(self, packet):
        entered = time.monotonic_ns()
        if self.task is None:
            self.task = asyncio.create_task(heartbeat(self.beats))
        value = packet.value
        reply = dict(item=value['item'], pid=os.getpid(), tid=threading.get_native_id(),
                     entered_ns=entered, input_stamps=packet.stamps,
                     dispatch_ns=value['dispatch_ns'], data_size=value['data'].size,
                     torch_imported='torch' in sys.modules, npu_imported='torch_npu' in sys.modules)
        reply['exit_ns'] = time.monotonic_ns()
        return Packet(reply, 'reply')

    async def inspect(self):
        return dict(pid=os.getpid(), beats=self.beats, torch_imported='torch' in sys.modules,
                    npu_imported='torch_npu' in sys.modules)


async def run(args, actors):
    records, beats = [], []
    beat_task = asyncio.create_task(heartbeat(beats))
    item = 0
    for block, mode in enumerate(args.modes.split(',')):
        previous, input_ref, entry = None, None, None

        async def finish_previous():
            nonlocal previous, input_ref, entry
            if previous is None:
                return
            entry['began_ns'] = time.monotonic_ns()
            futures, callbacks = [], []
            for rank, ref in enumerate(previous):
                f = ref.future()
                f.add_done_callback(lambda _, rank=rank: callbacks.append(dict(
                    rank=rank, ns=time.monotonic_ns(), tid=threading.get_native_id())))
                futures.append(asyncio.wrap_future(f))
            replies = await asyncio.gather(*futures)
            entry.update(done_ns=time.monotonic_ns(), callbacks=callbacks,
                         replies=[r.value | {'result_stamps': r.stamps} for r in replies])
            entry['wait_ms'] = (entry['done_ns']-entry['began_ns'])/1e6
            records.append(entry)
            if entry['wait_ms'] > 50:
                print('SLOW', item, mode, entry['wait_ms'], flush=True)
            if input_ref is not None:
                entry['free_start_ns'] = time.monotonic_ns()
                ray.internal.free(input_ref, local_only=False)
                entry['free_end_ns'] = time.monotonic_ns()
                input_ref = None

        for _ in range(args.iterations):
            await finish_previous()
            item += 1
            value = dict(item=item, dispatch_ns=time.monotonic_ns(),
                         data=np.arange(args.payload_bytes, dtype=np.uint8))
            packet = Packet(value, 'input')
            entry = dict(item=item, block=block, mode=mode, dispatch_ns=value['dispatch_ns'])
            if mode == 'put':
                input_ref = ray.put(packet)
                entry['put_end_ns'] = time.monotonic_ns()
                previous = [a.submit.remote(input_ref) for a in actors]
            elif mode == 'direct':
                previous = [a.submit.remote(packet) for a in actors]
            else:
                raise ValueError(mode)
            entry['submit_end_ns'] = time.monotonic_ns()
            await asyncio.sleep(args.interval_ms/1000)
        await finish_previous()
        print('BLOCK', block, mode, flush=True)
    snapshots = await asyncio.gather(*[a.inspect.remote() for a in actors])
    beat_task.cancel()
    return records, beats, snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=1500)
    p.add_argument('--interval-ms', type=float, default=12)
    p.add_argument('--payload-bytes', type=int, default=8192)
    p.add_argument('--modes', default='put,direct,direct,put')
    p.add_argument('--system-config', default='{}')
    args = p.parse_args()
    assert not args.output.exists(), args.output
    ctx = ray.init(num_cpus=4, num_gpus=0, include_dashboard=False,
                   _system_config=json.loads(args.system_config))
    session = Path(ray._private.worker._global_node.get_session_dir_path())
    print('SESSION', session, flush=True)
    try:
        actors = [Echo.remote() for _ in range(2)]
        ray.get([a.submit.remote(Packet(dict(item=0, dispatch_ns=time.monotonic_ns(),
                                            data=np.zeros(1, dtype=np.uint8)), 'input')) for a in actors])
        wall_anchor = dict(monotonic_ns=time.monotonic_ns(), time_ns=time.time_ns())
        records, beats, snapshots = asyncio.run(run(args, actors))
        data = dict(ray_version=ray.__version__, ray_commit=ray.__commit__, numpy_version=np.__version__,
                    cpu_arch=platform.machine(), python=sys.version, wall_anchor=wall_anchor,
                    environment={k:v for k,v in os.environ.items() if k.startswith(('RAY_', 'GRPC_'))},
                    session=str(session), options=vars(args) | {'output': str(args.output)},
                    torch_imported='torch' in sys.modules, npu_imported='torch_npu' in sys.modules,
                    records=records, beats=beats, workers=snapshots, complete=True)
        args.output.write_text(json.dumps(data, indent=2)+'\n')
        for mode in set(args.modes.split(',')):
            rows = [r for r in records if r['mode']==mode]
            print(mode, len(rows), 'max_wait_ms', max(r['wait_ms'] for r in rows),
                  'over50', sum(r['wait_ms']>50 for r in rows))
    finally:
        ray.shutdown()
        dest = args.output.parent.parent/'artifacts'/args.output.stem
        dest.mkdir(exist_ok=False)
        for src in (session/'logs').glob('*'):
            if src.is_file() and (src.name.startswith(('python-core-', 'raylet', 'gcs_server'))):
                shutil.copyfile(src, dest/src.name)


if __name__ == '__main__':
    main()
```
