# paired_norm_gc.py

```python
"""Same-worker, paired AB/BA full-model norm substitution experiment."""
import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--diagnostic-repeats', type=int, default=2)
    p.add_argument('--gc-policy', choices=('observe','disable-driver','disable-all'), default='observe')
    a = p.parse_args()
    from gc_observer import GCObserver
    driver_gc = GCObserver()
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
    from lmdeploy.pytorch.engine.executor import ray_executor
    from perf_worker_gc import PerfWorkerGC
    ray_executor.RayWorkerWrapper = PerfWorkerGC
    config = PytorchEngineConfig(device_type='ascend', dtype='bfloat16', tp=2,
        eager_mode=False, enable_prefix_caching=False, prefill_save_alignment=128, prefill_save_last_only=True, max_batch_size=8, session_len=8192,
        block_size=128, max_prefill_token_num=256, cache_max_entry_count=.5,
        language_model_only=True, logprobs_mode='raw_logprobs')
    data = {'options': vars(a), 'config': dataclasses.asdict(config), 'events': [], 'complete': False,
            'versions': {n:importlib.metadata.version(n) for n in ('torch','torch_npu','transformers')}}
    out = Path(a.output)
    def save():
        out.write_text(json.dumps(data, indent=2, default=str)+'\n')
    save()
    with pipeline('/afs-weights/Agents-A1', backend_config=config, trust_remote_code=True, log_level='INFO') as pipe:
        executor = pipe.async_engine.engine.executor
        data['workers'] = executor.collective_rpc('norm_perf_setup')
        if a.gc_policy != 'observe':
            import gc
            gc.disable()
        save()
        def run(bs, rep, mode, phase):
            state = executor.collective_rpc('norm_perf_mode', (mode, phase=='diagnostic', a.gc_policy=='disable-all'))
            prompts = [f'Trial {rep+2} item {i+1}. Explain how a computer stores and retrieves information. '
                       + 'Include examples of memory, disks, indexes, and cache. '*12 for i in range(bs)]
            rows = [dict(token_ids=[], times=[]) for _ in prompts]
            gen = GenerationConfig(max_new_tokens=256, do_sample=False, top_k=1,
                                   temperature=1., ignore_eos=True, random_seed=42)
            began = time.monotonic()
            for response in pipe.stream_infer(prompts, gen_config=gen):
                row = rows[response.index]
                if response.token_ids:
                    row['times'].append([time.monotonic()-began, len(response.token_ids)])
                    row['token_ids'].extend(response.token_ids)
                row['input_token_len'] = response.input_token_len
                row['finish_reason'] = response.finish_reason
                row['cached_tokens'] = response.cached_tokens
            ended = time.monotonic()
            elapsed = ended-began
            detail = executor.collective_rpc('norm_perf_collect')
            for row in rows:
                assert len(row['token_ids']) == 256, row
                assert row['finish_reason'] not in ('error','abort'), row
                row['ttft'] = row['times'][0][0]
                row['output_tps'] = (256-row['times'][0][1])/(row['times'][-1][0]-row['ttft'])
            event = dict(batch=bs, rep=rep, mode=mode, phase=phase, elapsed=elapsed,
                         throughput=bs*256/elapsed, responses=rows, prompts=prompts,
                         worker_before=state, worker_after=detail,
                         began=began, ended=ended, driver_gc=driver_gc.take())
            data['events'].append(event)
            save()
            print('PAIR', phase, bs, rep, mode, round(elapsed, 5), flush=True)
        for bs in (1,4):
            for mode in ('old','new'):
                run(bs, -2, mode, 'capture_warmup')
        for bs in (1,4):
            for mode in ('new','old'):
                run(bs, -1, mode, 'warmup')
        for bs in (1,4):
            for rep in range(a.repeats):
                for mode in (('old','new') if rep%2==0 else ('new','old')):
                    run(bs, rep, mode, 'measure')
        for bs in (1,4):
            for rep in range(a.diagnostic_repeats):
                for mode in (('old','new') if rep%2==0 else ('new','old')):
                    run(bs, rep, mode, 'diagnostic')
        data['token_comparison'] = []
        for phase in ('capture_warmup','warmup','measure','diagnostic'):
            groups = {}
            for event in data['events']:
                if event['phase']==phase:
                    groups.setdefault((event['batch'],event['rep']),{})[event['mode']]=event
            for key, pair in groups.items():
                equal = [r['token_ids'] for r in pair['old']['responses']] == [r['token_ids'] for r in pair['new']['responses']]
                data['token_comparison'].append(dict(phase=phase,batch=key[0],rep=key[1],equal=equal))
                save()
                if phase != 'capture_warmup':
                    assert equal, (phase,key)
        data['steady_paired_token_ids_equal'] = True
        data['complete'] = True
        save()
    print('SAVED', out, flush=True)


if __name__ == '__main__':
    main()
```
