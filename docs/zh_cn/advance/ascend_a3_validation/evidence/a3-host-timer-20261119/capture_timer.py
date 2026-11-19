#!/usr/bin/env python3
"""Read built-in timer/IRQ tracepoints for one CPU in a fresh private tracefs instance, or parse saved traces."""
import argparse
from collections import Counter,defaultdict
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid

REQUIRED=('irq/softirq_entry','irq/softirq_exit','timer/timer_expire_entry','timer/timer_expire_exit','irq/irq_handler_entry','irq/irq_handler_exit')
OPTIONAL='timer/timer_start'
LINE=re.compile(r'^\s*(.*?)\[(\d+)\]\s+[^:]*?\b(\d+\.\d+):\s+(?:(?:irq|timer):)?(softirq_entry|softirq_exit|timer_expire_entry|timer_expire_exit|timer_start|irq_handler_entry|irq_handler_exit):\s*(.*)$')
def field(body,key):
    m=re.search(r'(?:^|[\s\[])'+re.escape(key)+r'=([^\s,\]]+)',body)
    return m.group(1) if m else None
def function(body):
    m=re.search(r'(?:^|\s)function=(.+?)(?=\s+(?:now|baseclk|expires|flags|cpu|idx)=|$)',body)
    return m.group(1).strip() if m else None
def dump(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
def mask(cpu):
    words=[0]*(cpu//32+1);words[-1]=1<<(cpu%32)
    return ','.join(f'{w:08x}' for w in reversed(words))
def add_error(v,errors,k):
    errors[k]+=1
    if v is not None:v['valid']=False
def analyze(lines,cpu,loss=0,threshold_ms=100):
    arms={};errors=Counter();kinds=Counter();slow=[];current=None;last_t=-1;complete=0
    for line in lines:
        if re.search(r'\bLOST\s+\d+\s+EVENTS|PERF_RECORD_LOST',line,re.I):loss+=1
        header=re.search(r'entries-in-buffer/entries-written:\s*(\d+)/(\d+)',line)
        if header and int(header[2])>int(header[1]):loss+=1
        m=LINE.match(line)
        if not m or int(m[2])!=cpu:continue
        t=int(Decimal(m[3])*1_000_000_000);k=m[4];body=m[5];kinds[k]+=1
        if t<last_t:add_error(current,errors,'unordered_time')
        last_t=t
        if k=='timer_start':
            obj=field(body,'timer');fn=function(body)
            if obj and fn:arms[obj]={'ns':t,'function':fn,'timeout_ticks':field(body,'timeout'),'target_cpu':field(body,'cpu'),'flags':field(body,'flags')}
            continue
        if k in ('softirq_entry','softirq_exit'):
            if field(body,'vec')!='1':continue
            if k=='softirq_entry':
                if current is not None:add_error(current,errors,'vector_entry_before_exit')
                current=dict(start_ns=t,context=m[1].strip(),valid=True,open_cb=None,callbacks=[],irq_stack=[],irq_spans=[])
                continue
            if current is None:errors['unpaired_vector_exit']+=1;continue
            if current['open_cb'] or current['irq_stack']:add_error(current,errors,'unclosed_child')
            current['end_ns']=t;current['wall_ms']=(t-current['start_ns'])/1e6
            current['hardirq_ms']=sum((b-a)/1e6 for a,b in current['irq_spans'])
            current['callbacks_ms']=sum(v['wall_ms'] for v in current['callbacks'])
            current['unexplained_after_callbacks_and_known_irqs_ms']=current['wall_ms']-current['callbacks_ms']-current['hardirq_ms']+sum(v['nested_irq_ms'] for v in current['callbacks'])
            if current['unexplained_after_callbacks_and_known_irqs_ms']<-.005:add_error(current,errors,'overlap_exceeds_vector')
            if current['valid']:complete+=1
            if current['wall_ms']>=threshold_ms:slow.append(current)
            current=None;continue
        if current is None:continue
        if k=='irq_handler_entry':
            current['irq_stack'].append((field(body,'irq'),t))
        elif k=='irq_handler_exit':
            if not current['irq_stack'] or current['irq_stack'][-1][0]!=field(body,'irq'):
                add_error(current,errors,'irq_object_mismatch');continue
            _,a=current['irq_stack'].pop()
            if not current['irq_stack']:current['irq_spans'].append((a,t))
        elif k=='timer_expire_entry':
            obj=field(body,'timer');fn=function(body)
            if not obj or not fn or current['open_cb']:add_error(current,errors,'callback_entry_mismatch')
            prev=arms.pop(obj,None)
            current['open_cb']=dict(timer=obj,function=fn,start_ns=t,prior_local_arm=prev if prev and prev['function']==fn else None)
        elif k=='timer_expire_exit':
            cb=current['open_cb']
            if not cb or cb['timer']!=field(body,'timer'):
                add_error(current,errors,'callback_exit_mismatch');continue
            cb['end_ns']=t;cb['wall_ms']=(t-cb['start_ns'])/1e6
            cb['nested_irq_ms']=sum(max(0,min(t,b)-max(cb['start_ns'],a))/1e6 for a,b in current['irq_spans'])
            cb['wall_minus_known_hardirq_ms']=cb['wall_ms']-cb['nested_irq_ms']
            rearm=arms.get(cb['timer'])
            if rearm and cb['start_ns']<=rearm['ns']<=t and rearm['function']==cb['function']:cb['same_object_local_rearm']=rearm
            current['callbacks'].append(cb);current['open_cb']=None
    if current:errors['trailing_partial_vector']+=1
    stats=defaultdict(lambda:dict(calls=0,slow_vectors=0,sum_excluding_known_irq_ms=0.,max_one_call_excluding_known_irq_ms=0.,max_one_vector_sum_ms=0.,same_object_rearm_timeout_ticks=Counter()))
    if loss==0 and not errors['unordered_time']:
        for row in slow:
            if not row['valid']:continue
            by=Counter()
            for cb in row['callbacks']:
                f=cb['function'];s=stats[f];s['calls']+=1;u=cb['wall_minus_known_hardirq_ms'];by[f]+=u
                s['sum_excluding_known_irq_ms']+=u;s['max_one_call_excluding_known_irq_ms']=max(s['max_one_call_excluding_known_irq_ms'],u)
                if 'same_object_local_rearm' in cb:s['same_object_rearm_timeout_ticks'][str(cb['same_object_local_rearm']['timeout_ticks'])]+=1
            for f,total in by.items():stats[f]['slow_vectors']+=1;stats[f]['max_one_vector_sum_ms']=max(stats[f]['max_one_vector_sum_ms'],total)
    return dict(cpu=cpu,complete_vectors=complete,records=dict(kinds),loss_indicators=int(loss),errors=dict(errors),
                functions_in_complete_slow_vectors=dict(sorted(stats.items(),key=lambda x:-x[1]['sum_excluding_known_irq_ms'])),slow_vectors=slow)
def slurp(p):
    try:return p.read_text()
    except OSError:return None
def run():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cpu',type=int,required=True);ap.add_argument('--seconds',type=int,default=60)
    ap.add_argument('--out',type=Path);ap.add_argument('--tracefs',type=Path);ap.add_argument('--check',action='store_true')
    ap.add_argument('--parse',type=Path);ap.add_argument('--threshold-ms',type=float,default=100)
    args=ap.parse_args()
    if args.parse:
        if args.out is None:ap.error('--parse requires --out FILE')
        with args.parse.open() as f:result=analyze(f,args.cpu,threshold_ms=args.threshold_ms)
        dump(args.out,result);return 0
    if args.cpu<0 or args.seconds<=0 or args.threshold_ms<0:ap.error('invalid capture parameters')
    roots=[args.tracefs] if args.tracefs else list(map(Path,['/sys/kernel/tracing','/sys/kernel/debug/tracing']))
    root=next((p for p in roots if (p/'instances').is_dir()),None)
    status=dict(root=str(root) if root else None,can_create_private_instance=bool(root and os.access(root/'instances',os.W_OK)),
                events={n:bool(root and (root/'events'/n/'format').is_file()) for n in (*REQUIRED,OPTIONAL)})
    valid=status['can_create_private_instance'] and all(status['events'][n] for n in REQUIRED)
    if args.check or not valid:
        print(json.dumps(status,indent=2));return 0 if valid else 2
    if not args.out:ap.error('--out DIRECTORY is required')
    args.out.mkdir(parents=True,exist_ok=False)
    instance=root/'instances'/('a3-timer-'+uuid.uuid4().hex)
    instance.mkdir();out=args.out;buf=out/'trace.txt';stats={}
    try:
        (instance/'tracing_on').write_text('0')
        (instance/'tracing_cpumask').write_text(mask(args.cpu))
        (instance/'buffer_size_kb').write_text('8')
        (instance/f'per_cpu/cpu{args.cpu}/buffer_size_kb').write_text('8192')
        clock=instance/'trace_clock'
        if 'mono' in (slurp(clock) or '').replace('[','').replace(']','').split():clock.write_text('mono')
        for n in (*REQUIRED,OPTIONAL):
            if not status['events'][n]:continue
            node=instance/'events'/n
            if n.startswith('irq/softirq'):(node/'filter').write_text('vec == 1')
            (node/'enable').write_text('1')
        dump(out/'metadata.json',dict(cpu=args.cpu,seconds=args.seconds,clock=slurp(clock),private_instance=instance.name,
             events=status['events'],kernel=slurp(Path('/proc/sys/kernel/osrelease')),boot_id=slurp(Path('/proc/sys/kernel/random/boot_id'))))
        (instance/'tracing_on').write_text('1')
        try:time.sleep(args.seconds)
        finally:(instance/'tracing_on').write_text('0')
        cpu_root=instance/f'per_cpu/cpu{args.cpu}'
        rawstats=(cpu_root/'stats').read_text();(out/'buffer-stats.txt').write_text(rawstats)
        for s in rawstats.splitlines():
            if ':' in s:
                k,v=s.split(':',1);stats[k.strip()]=v.strip()
        # The snapshot is stopped. Read this CPU's ring only, then remove the instance we own.
        with (cpu_root/'trace').open('rb') as src,buf.open('xb') as dst:
            while True:
                data=src.read(1<<18)
                if not data:break
                dst.write(data)
    finally:
        try:(instance/'tracing_on').write_text('0');(instance/'events/enable').write_text('0')
        finally:instance.rmdir()
    loss=sum(int(stats.get(k,'0')) for k in ('overrun','dropped events','commit overrun'))
    with buf.open() as f:result=analyze(f,args.cpu,loss,args.threshold_ms)
    dump(out/'summary.json',result)
    print('complete TIMER vectors:',result['complete_vectors'],'slow:',len(result['slow_vectors']),'loss:',result['loss_indicators'])
    if not result['complete_vectors'] or result['loss_indicators']:return 3
    return 0
if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    try:raise SystemExit(run())
    except KeyboardInterrupt:raise SystemExit(130)
