#!/usr/bin/env python3
"""Two-phase, bounded host tasklet capture using owned tracefs facilities.

Run on the HOST as root, not by granting this container extra access.
Only an exclusively-created trace instance and uniquely-named kprobes are
changed. Existing tracers, IRQ affinities, sysctls and services are untouched.
Requires tracefs already mounted, readable BTF/kallsyms, CONFIG_KPROBE_EVENTS.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import threading
import time
import traceback

from host2_btf import tasklet_layout


def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def read(path):
    try:return Path(path).read_text()
    except OSError as e:return f'UNAVAILABLE: {e}'


def parse_symbols(text):
    symbols={}
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<3 or fields[1] not in ('t','T','w','W'):continue
        address=int(fields[0],16)
        if not address:continue
        name=fields[2];module=fields[3].strip('[]') if len(fields)>3 else None
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.$]*',name):continue
        symbols[address]=dict(name=name,module=module,probe_target=f'{module}:{name}' if module else name)
    return symbols


def discovery_definitions(group,layout):
    offsets=layout['offsets']
    fields=(f'tasklet=$arg1:x64 callback=+{offsets["func"]}($arg1):x64 '
            f'data=+{offsets["data"]}($arg1):x64 use_cb=+{offsets["use_callback"]}($arg1):u8')
    return [(f'tasklet_schedule',f'p:{group}/tasklet_schedule __tasklet_schedule {fields}'),
            (f'tasklet_hi_schedule',f'p:{group}/tasklet_hi_schedule __tasklet_hi_schedule {fields}')]


def reader(instance,path,stop,max_bytes):
    fd=os.open(instance/'trace_pipe',os.O_RDONLY|os.O_NONBLOCK)
    try:
        with path.open('xb',buffering=0) as out:
            size=0
            while not stop.is_set():
                try:data=os.read(fd,65536)
                except BlockingIOError:data=b''
                if data:
                    size+=len(data)
                    if size>max_bytes:raise RuntimeError('Trace byte limit reached; partial capture')
                    out.write(data)
                else:time.sleep(.01)
            while True:
                try:data=os.read(fd,65536)
                except BlockingIOError:break
                if not data:break
                out.write(data)
    finally:os.close(fd)


def monitor(path,stop,cpu):
    with path.open('x') as out:
        while not stop.wait(.02):
            row=dict(monotonic_ns=time.monotonic_ns(),cpu=cpu)
            row['cpu_stat']=next((l for l in read('/proc/stat').splitlines() if l.startswith(f'cpu{cpu} ')),None)
            row['softirqs']={}
            for line in read('/proc/softirqs').splitlines()[1:]:
                fields=line.split()
                if len(fields)>cpu+1:row['softirqs'][fields[0]]=int(fields[cpu+1])
            out.write(json.dumps(row)+'\n');out.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=int,default=18)
    p.add_argument('--seconds',type=float,default=40)
    p.add_argument('--discover-seconds',type=float,default=10)
    p.add_argument('--buffer-kb',type=int,default=8192)
    p.add_argument('--max-callbacks',type=int,default=16)
    p.add_argument('--max-trace-mb',type=int,default=512)
    p.add_argument('--tracefs',type=Path)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--perf-stacks',action='store_true',help='Also sample CPU kernel stacks at 49Hz; requires host perf permissions/tool')
    a=p.parse_args()
    assert 2<=a.discover_seconds<a.seconds<=300,'Require 2 <= discovery < total <= 300 seconds'
    assert 64<=a.buffer_kb<=65536
    assert 1<=a.max_callbacks<=64
    assert 1<=a.max_trace_mb<=4096
    a.output.mkdir(parents=True,exist_ok=False)
    metadata=dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),uname=list(platform.uname()),
        pid=os.getpid(),uid=os.geteuid(),options=vars(a)|{'output':str(a.output),'tracefs':str(a.tracefs)},
        time_anchor=dict(monotonic_ns=time.monotonic_ns(),time_ns=time.time_ns()),
        proc_status=read('/proc/self/status'),cpu_affinity=sorted(os.sched_getaffinity(0)),
        errors=[],cleanup_errors=[],created_probes=[],scope='Private tracefs instance + uniquely named probes only')
    candidates=[a.tracefs] if a.tracefs else [Path('/sys/kernel/tracing'),Path('/sys/kernel/debug/tracing')]
    tracefs=next((x for x in candidates if x and (x/'instances').is_dir()),None)
    if not tracefs:metadata['errors'].append('No mounted tracefs with instances. On host: mount -t tracefs tracefs /sys/kernel/tracing')
    elif not os.access(tracefs/'instances',os.W_OK) or not os.access(tracefs/'kprobe_events',os.W_OK):
        metadata['errors'].append('Host tracefs instance/kprobe creation is not writable with current permissions')
    metadata['tracefs']=str(tracefs)
    if a.perf_stacks and not shutil.which('perf'):metadata['errors'].append('--perf-stacks requested but perf executable is missing')
    try:metadata['tasklet_layout']=tasklet_layout()
    except Exception as e:metadata['errors'].append(f'Cannot verify tasklet layout from live BTF: {e}')
    kallsyms=read('/proc/kallsyms');symbols=parse_symbols(kallsyms)
    if not symbols:metadata['errors'].append('Kernel function addresses are unavailable; cannot resolve callbacks safely')
    (a.output/'kallsyms.txt').write_text(kallsyms)
    for filename,source in [('modules-before.txt','/proc/modules'),('interrupts-before.txt','/proc/interrupts'),
                            ('softirqs-before.txt','/proc/softirqs'),('cpu-before.txt','/proc/stat')]:
        (a.output/filename).write_text(read(source))
    group=f'a3h2_{os.getpid()}_{time.monotonic_ns()}'
    if 'tasklet_layout' in metadata:
        metadata['proposed_discovery_probes']=discovery_definitions(group,metadata['tasklet_layout'])
    dump(a.output/'preflight.json',metadata)
    if a.check_only or metadata['errors']:
        print(json.dumps(dict(ready=not metadata['errors'],errors=metadata['errors'],output=str(a.output)),indent=2))
        return 0 if not metadata['errors'] else 2
    instance=tracefs/'instances'/group
    stop=threading.Event();threads=[];started=False
    created=[];callbacks={};perf=None;perf_log=None
    def write(path,text):path.write_text(str(text)+'\n')
    def register(event,definition):
        with (tracefs/'kprobe_events').open('a') as f:f.write(definition+'\n')
        created.append(event);metadata['created_probes'].append(definition)
        source=instance/'events'/group/event
        write(source/'enable',1)
        (a.output/f'format-{event}.txt').write_text(read(source/'format'))
    def on_signal(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    previous={s:signal.signal(s,on_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        # Keep the collector runnable while CPU18 itself is stalled.
        mask=os.sched_getaffinity(0)-{a.cpu}
        if mask:os.sched_setaffinity(0,mask)
        instance.mkdir();started=True
        write(instance/'tracing_on',0);write(instance/'current_tracer','nop')
        clocks=(instance/'trace_clock').read_text().replace('[','').replace(']','').split()
        if 'mono' not in clocks:raise RuntimeError('CLOCK_MONOTONIC trace clock is required for cross-log alignment')
        write(instance/'trace_clock','mono')
        write(instance/'tracing_cpumask',format(1<<a.cpu,'x'))
        write(instance/'buffer_size_kb',8)
        write(instance/f'per_cpu/cpu{a.cpu}/buffer_size_kb',a.buffer_kb)
        for event in ['irq/softirq_raise','irq/softirq_entry','irq/softirq_exit',
                      'irq/irq_handler_entry','irq/irq_handler_exit','sched/sched_switch']:
            target=instance/'events'/event
            if not target.exists():raise RuntimeError(f'Missing required tracepoint {event}')
            write(target/'enable',1)
            (a.output/f'format-{event.replace("/","-")}.txt').write_text(read(target/'format'))
        for event,definition in discovery_definitions(group,metadata['tasklet_layout']):register(event,definition)
        if a.perf_stacks:
            command=['perf','record','-a','-C',str(a.cpu),'-F','49','-g','-e','cycles:k','-o',str(a.output/'perf.data'),'--','sleep',str(a.seconds)]
            metadata['perf_command']=command
            perf_log=(a.output/'perf-record.log').open('w')
            perf=subprocess.Popen(command,stdout=perf_log,stderr=subprocess.STDOUT,start_new_session=True)
        def guarded(fn,args):
            try:fn(*args)
            except BaseException as e:
                metadata['errors'].append(f'{fn.__name__} failed: {e}');stop.set()
                try:write(instance/'tracing_on',0)
                except OSError:pass
        for fn,args in [(reader,(instance,a.output/'trace.txt',stop,a.max_trace_mb*1024*1024)),(monitor,(a.output/'cpu-softirq.jsonl',stop,a.cpu))]:
            thread=threading.Thread(target=guarded,args=(fn,args),daemon=True);thread.start();threads.append(thread)
        write(instance/'trace_marker','HOST2_DISCOVERY_BEGIN')
        write(instance/'tracing_on',1);time.sleep(a.discover_seconds);write(instance/'tracing_on',0)
        time.sleep(.1)
        if stop.is_set():raise RuntimeError('Reader/monitor failed during discovery')
        trace=(a.output/'trace.txt').read_text()
        addresses=sorted({int(x,16) for x in re.findall(r'callback=((?:0x)?[0-9a-fA-F]+)',trace)})
        metadata['discovered_addresses']=[hex(x) for x in addresses]
        if len(addresses)>a.max_callbacks:
            raise RuntimeError(f'{len(addresses)} callback addresses exceed --max-callbacks={a.max_callbacks}; review discovery before increasing the limit')
        unresolved=[]
        for addr in addresses:
            if addr not in symbols:unresolved.append(hex(addr));continue
            name=f'cb{len(callbacks):03d}';s=symbols[addr]
            record=s|dict(address=hex(addr),event=name)
            callbacks[name]=record
            try:
                register(name+'_entry',f'p:{group}/{name}_entry {s["probe_target"]} arg0=$arg1:x64')
                register(name+'_exit',f'r128:{group}/{name}_exit {s["probe_target"]}')
                record['attached']=True
            except OSError as e:
                record['attached']=False;record['error']=str(e)
                metadata['errors'].append(f'Cannot attach callback {s["probe_target"]}: {e}')
        metadata['unresolved_addresses']=unresolved
        if unresolved:metadata['errors'].append('Some callback addresses cannot be resolved exactly')
        if not callbacks:metadata['errors'].append('No tasklet callback discovered; collection does not identify a callback')
        dump(a.output/'callbacks.json',callbacks)
        write(instance/'trace_marker','HOST2_CALLBACK_TIMING_BEGIN')
        write(instance/'tracing_on',1);time.sleep(a.seconds-a.discover_seconds);write(instance/'tracing_on',0)
        metadata['buffer_stats']=read(instance/f'per_cpu/cpu{a.cpu}/stats')
        metadata['probe_profile']='\n'.join(l for l in read(tracefs/'kprobe_profile').splitlines() if group in l)
        metadata['capture_finished']=True
    except BaseException as e:
        metadata['errors'].append(f'{type(e).__name__}: {e}')
        metadata['traceback']=traceback.format_exc()
    finally:
        if started:
            try:write(instance/'tracing_on',0);write(instance/'events/enable',0)
            except OSError as e:metadata['cleanup_errors'].append(str(e))
        stop.set()
        for thread in threads:thread.join(timeout=5)
        if perf is not None:
            if perf.poll() is None:os.killpg(perf.pid,signal.SIGINT)
            try:metadata['perf_returncode']=perf.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(perf.pid,signal.SIGKILL);perf.wait();metadata['errors'].append('perf did not stop within timeout')
            perf_log.close()
            if metadata.get('perf_returncode') not in (0,130):metadata['errors'].append('perf record failed; inspect perf-record.log')
            if (a.output/'perf.data').is_file():
                with (a.output/'perf-script.txt').open('w') as out:
                    result=subprocess.run(['perf','script','-i',str(a.output/'perf.data')],stdout=out,stderr=subprocess.PIPE,text=True)
                (a.output/'perf-script.err').write_text(result.stderr)
                if result.returncode:metadata['errors'].append('perf script failed to symbolize stacks')
        for event in reversed(created):
            try:
                with (tracefs/'kprobe_events').open('a') as f:f.write(f'-:{group}/{event}\n')
            except OSError as e:metadata['cleanup_errors'].append(f'{group}/{event}: {e}')
        if started:
            try:instance.rmdir()
            except OSError as e:metadata['cleanup_errors'].append(f'Private instance {instance}: {e}')
        for sig,handler in previous.items():signal.signal(sig,handler)
        for filename,source in [('modules-after.txt','/proc/modules'),('interrupts-after.txt','/proc/interrupts'),
                                ('softirqs-after.txt','/proc/softirqs'),('cpu-after.txt','/proc/stat')]:
            (a.output/filename).write_text(read(source))
        metadata['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata['recorded_files']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in a.output.iterdir() if p.is_file() and p.name not in ('metadata.json',)}
        dump(a.output/'metadata.json',metadata)
    print(json.dumps(dict(callbacks=callbacks,errors=metadata['errors'],cleanup_errors=metadata['cleanup_errors']),indent=2))
    return 0 if not metadata['errors'] and not metadata['cleanup_errors'] else 3


if __name__=='__main__':raise SystemExit(main())
