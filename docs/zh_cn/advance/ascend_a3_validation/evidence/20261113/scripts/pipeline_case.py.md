# pipeline_case.py

原始验证文件快照；源码 SHA256：`39eb448c5016d39c2615802d5555c1c61b34742e0af9f718668053afc4b083c8`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Matched baseline/native generation, trace, and streaming performance harness."""
import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/afs-weights/Sage-A1.5')
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--dp', type=int, default=1)
    p.add_argument('--ep', type=int, default=1)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--mtp', type=int, default=0)
    p.add_argument('--prefill', type=int, default=8192)
    p.add_argument('--session', type=int, default=262144)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--cache-fraction', type=float, default=.95)
    p.add_argument('--scores', action='store_true')
    p.add_argument('--no-thinking', action='store_true')
    p.add_argument('--respect-eos', action='store_true')
    p.add_argument('--skip-input-scoring', action='store_true')
    p.add_argument('--kv-config')
    p.add_argument('--new-tokens', type=int, default=64)
    p.add_argument('--suite', choices=['smoke', 'trace', 'bench', 'long', 'full', 'extra'], default='trace')
    p.add_argument('--length', type=int, default=8192)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--bench-concurrency', type=int, nargs='+', default=[1,4,16])
    p.add_argument('--bench-http-workload', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
    from lmdeploy.messages import SpeculativeConfig
    config = PytorchEngineConfig(device_type='ascend', dtype='bfloat16', tp=a.tp,
        dp=a.dp, ep=a.ep, eager_mode=a.eager, enable_prefix_caching=a.prefix,
        prefix_cache_state_budget=16 if a.prefix else 0, prefill_save_alignment=128,
        prefill_save_last_only=True, max_batch_size=a.batch, session_len=a.session,
        block_size=128, max_prefill_token_num=a.prefill, cache_max_entry_count=a.cache_fraction,
        language_model_only=True, logprobs_mode='raw_logprobs' if a.scores else None)
    spec = SpeculativeConfig(method='qwen3_5_mtp', num_speculative_tokens=a.mtp) if a.mtp else None
    if a.kv_config:
        from lmdeploy.messages import KVTransferConfig
        config.kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector',kv_role='kv_both',
            kv_connector_extra_config={'mooncake_config_path':str(Path(a.kv_config).resolve()),
                                       'device_backend':'ascend',
                                       'model_name':Path(a.model).name})
    data = dict(options=vars(a), engine_config=dataclasses.asdict(config),
                environment=os.environ.get('VALIDATION_ENV'), events=[], complete=False,
                versions={n:importlib.metadata.version(n) for n in ['torch','torch_npu','transformers']})
    out = Path(a.output)
    def save():
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    save()
    def generation(tokens=None, logprobs=5):
        return GenerationConfig(max_new_tokens=tokens or a.new_tokens, do_sample=False,
            top_k=1, temperature=1., ignore_eos=not a.respect_eos, random_seed=42, logprobs=logprobs)
    prompts = [
        'Compute 17 * 23. Give the result and a brief explanation.',
        '请用一句话解释为什么天空是蓝色的。',
        'Write a Python function that returns the sum of even numbers in a list.',
        'Translate to English: 高效推理需要谨慎管理内存。',
        'A train travels 120 km in 2 hours. What is its average speed?',
        'List the first ten prime numbers in increasing order.',
        'Explain the difference between a stack and a queue with one example.',
        'The secret access code is MAPLE-7391. Repeat only the secret access code.',
    ]
    started = time.monotonic()
    with pipeline(a.model, backend_config=config, speculative_config=spec,
                  trust_remote_code=True, log_level='INFO') as pipe:
        data['load_seconds'] = time.monotonic()-started
        def infer(label, batch, tokens=None):
            t = time.monotonic()
            extra = {'chat_template_kwargs': {'enable_thinking': False}} if a.no_thinking else {}
            responses = pipe(batch, gen_config=generation(tokens), **extra)
            rows = [dict(token_ids=r.token_ids, text=r.text, logprobs=r.logprobs,
                input_token_len=r.input_token_len, generate_token_len=r.generate_token_len,
                cached_tokens=r.cached_tokens, finish_reason=r.finish_reason) for r in responses]
            data['events'].append(dict(label=label,prompts=batch,responses=rows,elapsed=time.monotonic()-t))
            save()
            for row in rows:
                assert row['finish_reason'] not in ('error','abort'), row
                if a.respect_eos:
                    assert 0 < len(row['token_ids']) <= (tokens or a.new_tokens), row
                else:
                    assert len(row['token_ids']) == (tokens or a.new_tokens), row
                assert all(math.isfinite(v) for d in row['logprobs'] or [] for v in d.values()), row
            print('CASE',label,'OK',time.monotonic()-t,flush=True)
            return rows
        if a.suite == 'smoke':
            infer('smoke', prompts[:1])
        elif a.suite in ('trace','full'):
            first = infer('single-before',prompts[:1])
            infer('batch-eight',prompts)
            tail = 'Read the following notes carefully. ' + 'One two three four five. ' * 180
            infer('mixed-long-short',[tail+' Summarize in one sentence.',prompts[1], 'Hi.'])
            again = infer('single-after',prompts[:1])
            data['repeat_exact'] = first[0]['token_ids']==again[0]['token_ids']
            # Same batch shape as the earlier multi-request call.
            infer('batch-eight-repeat',prompts)
        elif a.suite == 'long':
            tok = pipe.async_engine.tokenizer
            filler = 'The archive contains ordinary records of weather and travel. '
            needle = '\nThe unique verification code for this archive is ORCHID-8246.\n'
            # Repeated text merges BPE tokens at boundaries; measure its marginal
            # token count rather than multiplying the isolated sentence length.
            per_repeat = len(tok.encode(filler*101, add_bos=False))-len(tok.encode(filler*100, add_bos=False))
            text = filler * max(1,(a.length-200)//per_repeat)
            mid = len(text)//2
            prompt = text[:mid]+needle+text[mid:]+'\nWhat is the unique verification code? Reply with only the code.'
            rows = infer('long-retrieval',[prompt])
            data['retrieval_contains_code'] = 'ORCHID-8246' in rows[0]['text']
        elif a.suite == 'extra':
            from extra_cases import run_extra
            run_extra(pipe, data, save, a)
        if a.suite in ('bench','full'):
            for bs in a.bench_concurrency:
                if bs>a.batch: continue
                for rep in range(-1,a.repeats):
                    # Distinct prefix prevents previous repetitions from becoming cache hits.
                    batch = [f'Trial {rep+2} item {i+1}. Explain how a computer stores and retrieves information. '
                             + 'Include examples of memory, disks, indexes, and cache. '*12 for i in range(bs)]
                    if a.bench_http_workload:
                        batch=[f'Workload C{bs} R{rep+2} Q{i}. Explain how computers store information. '
                            + 'Discuss memory, disk, files, and cache with examples. '*12 for i in range(bs)]
                    streams=[dict(token_ids=[], logprobs=[], times=[],text='',cached_tokens=0) for _ in batch]
                    t=time.monotonic()
                    stream_kwargs={'chat_template_kwargs':{'enable_thinking':False}} if a.no_thinking else {}
                    for r in pipe.stream_infer(batch,gen_config=generation(logprobs=None),**stream_kwargs):
                        row=streams[r.index]
                        now=time.monotonic()-t
                        if r.token_ids:
                            row['times'].append([now,len(r.token_ids)])
                            row['token_ids'].extend(r.token_ids)
                        row['text']+=r.text
                        row['input_token_len']=r.input_token_len
                        row['finish_reason']=r.finish_reason
                        row['cached_tokens']=r.cached_tokens
                    elapsed=time.monotonic()-t
                    for row in streams:
                        assert len(row['token_ids'])==a.new_tokens, row
                        assert row['finish_reason'] not in ('error','abort'),row
                        row['ttft']=row['times'][0][0]
                        row['output_tps']=(len(row['token_ids'])-row['times'][0][1])/max(1e-9,row['times'][-1][0]-row['ttft'])
                        row['slo_pass']=row['ttft']<=3 and row['output_tps']>=40
                    data['events'].append(dict(label=f'bench-c{bs}-r{rep}',warmup=rep<0,batch=bs,
                        elapsed=elapsed,throughput=bs*a.new_tokens/elapsed,responses=streams,prompts=batch))
                    save()
                    print('BENCH',bs,rep,elapsed,flush=True)
        data['complete']=True
        data['dlinfer_modules']=[n for n in sys.modules if n.split('.')[0]=='dlinfer']
        if os.environ.get('VALIDATION_ENV','').startswith('native'):
            assert not data['dlinfer_modules']
            assert not any('dlinfer' in d.metadata['Name'].lower() for d in importlib.metadata.distributions())
        save()
    print('SAVED',out,flush=True)

if __name__=='__main__':
    main()

```
