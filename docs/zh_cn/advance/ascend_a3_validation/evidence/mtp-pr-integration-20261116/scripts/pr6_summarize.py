"""Verify the integrated PR against the saved correct MTP reference."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pr6-summary-r0.json'
assert not OUT.exists()
HEAD = '85a0581228ad33a695430a1f43a47358a8258131'

def read(folder,label): return json.loads((ROOT/folder/(label+'.json')).read_text())

def bit_equal(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,float): return struct.pack('!d',a)==struct.pack('!d',b)
    if isinstance(a,list): return len(a)==len(b) and all(bit_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,dict): return a.keys()==b.keys() and all(bit_equal(a[k],b[k]) for k in a)
    return a==b

reference=read('results','mtp-stage1-suite-r0')
rows=[]
for profile in ('old','new'):
    for repeat in (0,1):
        label=f'pr6-{profile}-mtp-r{repeat}'
        log=read('logs',label);d=read('results',label)
        assert log['returncode']==0 and log['source_head']==HEAD and d['complete']
        assert not d['dlinfer_modules']
        assert log['diff_sha256']==hashlib.sha256(b'').hexdigest()
        assert len(d['events'])==len(reference['events'])
        pairs=[]
        for event,ref in zip(d['events'],reference['events']):
            assert event['label']==ref['label'] and event['prompts']==ref['prompts']
            assert len(event['responses'])==len(ref['responses'])
            pairs.extend(zip(event['responses'],ref['responses']))
        assert len(pairs)==21
        row=dict(label=label,responses=len(pairs),
                 token_sequences_exact=sum(bit_equal(x['token_ids'],y['token_ids']) for x,y in pairs),
                 top5_scores_bitwise_exact=sum(bit_equal(x['logprobs'],y['logprobs']) for x,y in pairs),
                 generated_tokens=sum(len(x['token_ids']) for x,y in pairs),
                 recorded_score_values=sum(len(scores) for x,y in pairs for scores in x['logprobs']),
                 source_head=log['source_head'],dlinfer_modules=d['dlinfer_modules'])
        assert row['token_sequences_exact']==row['top5_scores_bitwise_exact']==21
        rows.append(row)
for label in ('pr6-old-hw-r0','pr6-new-hw-r0','pr6-cpu-r0'):
    log=read('logs',label);assert log['returncode']==0 and log['source_head']==HEAD
    if 'hw' in label:
        text=(ROOT/'logs'/(label+'.log')).read_text()
        assert '6 passed' in text and 'OBSERVED_CIRCULAR_NUM_STAGES [1]' in text
data=dict(source_head=HEAD,reference='results/mtp-stage1-suite-r0.json',
          policy='num_stages = 1 if cache_seqlens_rb is not None else 3',
          hardware={'old_npuir_1_2_042c923d':6,'new_npuir_2_0_efb71415':6},
          cpu={'passed':32,'skipped':1},mtp=rows,
          compiler_profile='Optional triton-ascend==3.2.2.post2 for an existing compatible runtime; legacy Docker unchanged.',
          performance='Previous measured results retained with their original source/compiler attribution; no new performance benchmark in this integration.',
          limitations=['CPU18 exact callback still needs privileged host capture; no affinity or Ray RPC order change shipped.',
                       'No full Docker/CANN9.2 runtime upgrade certification or new HTTP throughput result.',
                       'NVIDIA CI/full-model coverage remains attributed to its actual tested commit.'])
OUT.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
