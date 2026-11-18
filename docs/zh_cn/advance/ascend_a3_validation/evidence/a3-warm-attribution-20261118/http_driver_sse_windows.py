"""Match retained client SSE and engine output windows using the decoded original input."""
import json
from pathlib import Path
from statistics import median
from tokenizers import Tokenizer

root=Path(__file__).resolve().parents[1]
out={}
for label in ('agents-historical-r0','agents-current-r0','sage-current-r0'):
    cache={}
    records=[]
    blocks=[]
    for block in range(8):
        modes={}
        for mode in ('old','new'):
            d=json.loads((root/'results'/f'{label}-{mode}-b{block:02d}.json').read_text())
            tok=cache.get('tok')
            if tok is None:
                tok=Tokenizer.from_file(d['options']['model']+'/tokenizer.json')
                cache['tok']=tok
            traces={}
            for t in d['token_traces']:
                if d['phases'][str(t['phase'])]['kind']!='measure':continue
                inp=tuple(t['input_ids'])
                if inp not in cache:cache[inp]=tok.decode(inp,skip_special_tokens=False)
                traces.setdefault(t['phase'],[]).append((cache[inp],t))
            keyed={}
            for e in d['events']:
                if e['kind']!='measure':continue
                phase_traces=traces.pop(e['phase'])
                for i,r in enumerate(e['responses']):
                    prompt=r['prompt'].strip()
                    matches=[t for decoded,t in phase_traces if prompt in decoded]
                    assert len(matches)==1
                    t=matches[0]
                    driver=(t['chunks'][-1][0]-t['chunks'][0][0])/1e6
                    sse=(r['events'][-1]['time']-r['events'][0]['time'])*1000
                    assert abs(sse-127000/r['output_tps'])<1e-6
                    key=(e['batch'],e['rep'],i)
                    keyed[key]=(prompt,driver,sse,t['token_ids'])
            assert not traces and len(keyed)==57
            modes[mode]=keyed
        rows=[]
        for key,old in modes['old'].items():
            new=modes['new'][key]
            assert old[0]==new[0] and old[3]==new[3]
            row=dict(block=block,batch=key[0],rep=key[1],index=key[2],old_driver_ms=old[1],
                     new_driver_ms=new[1],old_sse_ms=old[2],new_sse_ms=new[2],
                     delta_driver_ms=new[1]-old[1],delta_sse_ms=new[2]-old[2],
                     delta_window_difference_ms=(new[2]-old[2])-(new[1]-old[1]))
            rows.append(row)
        records.extend(rows)
        for bs in (1,4):
            subset=[r for r in rows if r['batch']==bs]
            blocks.append(dict(block=block,batch=bs,**{k:median(r[k] for r in subset) for k in ('delta_driver_ms','delta_sse_ms','delta_window_difference_ms')}))
    out[label]=dict(blocks=blocks,requests=records)
    for bs in (1,4):
        print(label,bs,{k:round(median(r[k] for r in blocks if r['batch']==bs),2) for k in ('delta_driver_ms','delta_sse_ms','delta_window_difference_ms')})
(root/'results'/'http-driver-sse-windows-r0.json').write_text(json.dumps(out,indent=2)+'\n')
