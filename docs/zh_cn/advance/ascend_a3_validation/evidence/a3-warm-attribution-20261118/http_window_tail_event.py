"""Inspect all four matched server/client streams at the largest window residual."""
import json
from pathlib import Path
from tokenizers import Tokenizer

root=Path(__file__).resolve().parents[1]
windows=json.loads((root/'results/http-driver-sse-windows-r0.json').read_text())
largest=max(((lab,r) for lab,res in windows.items() for r in res['requests']),key=lambda p:abs(p[1]['delta_window_difference_ms']))
label,row=largest
out={'label':label,'largest_residual':row,'requests':[]}
for mode in ('old','new'):
    d=json.loads((root/'results'/f'{label}-{mode}-b{row["block"]:02d}.json').read_text())
    tok=Tokenizer.from_file(d['options']['model']+'/tokenizer.json')
    e=next(e for e in d['events'] if e['kind']=='measure' and e['rep']==row['rep'] and e['batch']==row['batch'])
    for i,r in enumerate(e['responses']):
        t=next(t for t in d['token_traces'] if t['phase']==e['phase'] and r['prompt'].strip() in tok.decode(t['input_ids']))
        texts=[s['content']+s['reasoning'] for s in r['events']]
        assert len(texts)==len(t['chunks'])==len(t['token_ids'])==128
        assert all(n==1 for _,n in t['chunks'])
        assert all(tok.decode([idx],skip_special_tokens=False)==txt for idx,txt in zip(t['token_ids'],texts))
        gaps=[dict(next_token=j+1,driver_ms=(t['chunks'][j][0]-t['chunks'][j-1][0])/1e6,
                   sse_ms=(r['events'][j]['time']-r['events'][j-1]['time'])*1000) for j in range(1,128)]
        out['requests'].append(dict(mode=mode,index=i,gaps=gaps))
(root/'results/http-largest-window-residual-r0.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out['requests']:
    if r['mode']=='new':print(r['index'],r['gaps'][117])
