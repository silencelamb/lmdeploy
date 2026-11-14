# extra_cases.py

原始验证文件快照；源码 SHA256：`7b92d10d4e4e62dc62162411db2399c39f68d27c276808465c6b2e6a93db5648`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Reference answers, fixed-history scoring, real prefix hits, and long retrieval."""
import math
import time
from lmdeploy import GenerationConfig

def run_extra(pipe,data,save,args):
    def run(label,prompts,config):
        start=time.monotonic()
        rs=pipe(prompts,gen_config=config,chat_template_kwargs={'enable_thinking':False})
        rows=[dict(token_ids=r.token_ids,text=r.text,logprobs=r.logprobs,
            input_token_len=r.input_token_len,generate_token_len=r.generate_token_len,
            cached_tokens=r.cached_tokens,finish_reason=r.finish_reason,
            error_message=r.error_message) for r in rs]
        data['events'].append(dict(label=label,prompts=prompts,responses=rows,elapsed=time.monotonic()-start))
        save()
        assert all(r['finish_reason'] not in ('error','abort') for r in rows), rows
        print('EXTRA',label,'OK',time.monotonic()-start,flush=True)
        return rows
    quality=[
        ('17 * 23','391'),('125 + 378','503'),('1024 / 16','64'),('19 * 19','361'),
        ('2 ** 10','1024'),('81 - 27','54'),('the next number in 2, 4, 8, 16','32'),
        ('the number of days in a non-leap year','365'),
        ('the number of minutes in 3 hours','180'),('the greatest common divisor of 18 and 24','6'),
        ('三十七加四十八','85'),('一百减去二十三','77'),
    ]
    gen=GenerationConfig(max_new_tokens=32,do_sample=False,top_k=1,random_seed=42,logprobs=5)
    qs=[f'What is {q}? Reply with only the integer answer, no explanation.' for q,_ in quality]
    rows=run('exact-answer',qs,gen)
    data['quality']=[dict(question=q,expected=expected,actual=r['text'].strip(),
                         correct=r['text'].strip()==expected) for (q,expected),r in zip(quality,rows)]
    save()
    # Scores use the same fixed teacher-forced history on both branches, so
    # numerical comparison is meaningful even if generated continuations differ.
    scoring=[f'Question: {q}. The correct answer is {expected}.' for q,expected in quality[:4]]
    cfg=GenerationConfig(max_new_tokens=0,logprobs=5,logprob_start_len=0,do_sample=False,top_k=1)
    if not args.skip_input_scoring:
        scored=run('fixed-history-scores',scoring,cfg)
        data['scoring_rows_present']=all(bool(r['logprobs']) for r in scored)
        save()
    long_text='Archive record: the weather was mild and the road was open. '*max(1,(args.length-160)//14)
    mid=len(long_text)//2
    needle=' The unique verification code is ORCHID-8246. '
    prompt=long_text[:mid]+needle+long_text[mid:]+' What is the unique verification code? Reply only with the code.'
    long_cfg=GenerationConfig(max_new_tokens=32,do_sample=False,top_k=1,random_seed=42,logprobs=5)
    first=run('prefix-long-first',[prompt],long_cfg)
    run('state-distractor',['Repeat only: MOUNTAIN-5173','Compute 9 * 9. Reply only with the integer.'],gen)
    again=run('prefix-long-repeat',[prompt],long_cfg)
    data['prefix_check']=dict(exact_tokens=first[0]['token_ids']==again[0]['token_ids'],
        first_cached=first[0]['cached_tokens'],repeat_cached=again[0]['cached_tokens'],
        input_tokens=first[0]['input_token_len'],
        retrieval_first='ORCHID-8246' in first[0]['text'],retrieval_repeat='ORCHID-8246' in again[0]['text'])
    save()

```
