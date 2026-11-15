"""Minimal read-only BTF layout reader; no dependency on host kernel headers."""
import hashlib
import json
from pathlib import Path
import struct


def tasklet_layout(path='/sys/kernel/btf/vmlinux'):
    raw=Path(path).read_bytes()
    order='<' if raw[:2]==b'\x9f\xeb' else '>'
    magic,version,flags,hdr_len,toff,tlen,soff,slen=struct.unpack_from(order+'HBBIIIII',raw)
    if magic!=0xeb9f or version!=1:raise ValueError('Unsupported BTF header')
    strings=raw[hdr_len+soff:hdr_len+soff+slen]
    def name(offset):return strings[offset:strings.find(b'\0',offset)].decode()
    types={};pos=hdr_len+toff;end=pos+tlen;tid=1
    while pos<end:
        no,info,size=struct.unpack_from(order+'III',raw,pos);pos+=12
        kind=(info>>24)&31;vlen=info&0xffff;kflag=info>>31
        item=dict(name=name(no),kind=kind,size_or_type=size)
        if kind in (4,5):
            members=[]
            for i in range(vlen):
                mn,mt,mo=struct.unpack_from(order+'III',raw,pos+12*i)
                members.append(dict(name=name(mn),type=mt,bit_offset=mo&0xffffff if kflag else mo,
                                    bitfield_size=mo>>24 if kflag else 0))
            item['members']=members
        lengths={0:0,1:4,2:0,3:12,4:12*vlen,5:12*vlen,6:8*vlen,7:0,8:0,9:0,
                 10:0,11:0,12:0,13:8*vlen,14:4,15:12*vlen,16:0,17:4,18:0,19:12*vlen}
        if kind not in lengths:raise ValueError(f'Unknown BTF kind {kind}')
        pos+=lengths[kind];types[tid]=item;tid+=1
    roots=[i for i,t in types.items() if t['name']=='tasklet_struct' and t['kind']==4]
    if len(roots)!=1:raise ValueError(f'Expected one tasklet_struct, got {roots}')
    offsets={}
    def visit(tid,base=0):
        t=types[tid]
        if t['kind'] in (8,9,10,11,18):return visit(t['size_or_type'],base)
        for m in t.get('members',[]):
            if m['bit_offset']%8 or m['bitfield_size'] not in (0,8):
                raise ValueError('Unsupported tasklet bitfield')
            off=base+m['bit_offset']//8
            if m['name']:offsets[m['name']]=off
            else:visit(m['type'],off)
    visit(roots[0])
    for field in ('func','callback','use_callback','data'):
        if field not in offsets:raise ValueError(f'Missing tasklet field {field}')
    if offsets['func']!=offsets['callback']:raise ValueError('Unexpected callback union')
    return dict(btf_path=path,btf_sha256=hashlib.sha256(raw).hexdigest(),type_id=roots[0],
                size=types[roots[0]]['size_or_type'],offsets=offsets)


if __name__=='__main__':print(json.dumps(tasklet_layout(),indent=2))
