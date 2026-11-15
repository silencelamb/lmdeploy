# profile_pipeline.py

```python
"""Original model and norms; only validation stage timers and optional stacks."""
import dataclasses
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
    from lmdeploy.pytorch.engine.executor import ray_executor
    import stage_probe
    ray_executor.RayWorkerWrapper = stage_probe.StageWorker
    config = PytorchEngineConfig(device_type='ascend', dtype='bfloat16', tp=2,
        eager_mode=False, enable_prefix_caching=False, prefill_save_alignment=128,
        prefill_save_last_only=True, max_batch_size=8, session_len=8192, block_size=128,
        max_prefill_token_num=256, cache_max_entry_count=.5, language_model_only=True,
        logprobs_mode='raw_logprobs')
    data = dict(config=dataclasses.asdict(config), events=[], complete=False)
    out = ROOT/'results/agents-stage-profile.json'
    def save():
        out.write_text(json.dumps(data,indent=2,default=str)+'\n')
    save()
    with pipeline('/afs-weights/Agents-A1', backend_config=config, trust_remote_code=True, log_level='INFO') as pipe:
        executor = pipe.async_engine.engine.executor
        data['workers'] = executor.collective_rpc('stage_setup')
        data['driver'] = stage_probe.setup_driver()
        for bs in (1,4):
            for rep in range(-1,8):
                prompts = [f'Trial {rep+2} item {i+1}. Explain how a computer stores and retrieves information. '
                           + 'Include examples of memory, disks, indexes, and cache. '*12 for i in range(bs)]
                rows=[dict(token_ids=[],times=[]) for _ in prompts]
                began=time.monotonic()
                gen=GenerationConfig(max_new_tokens=256,do_sample=False,top_k=1,temperature=1.,ignore_eos=True,random_seed=42)
                for response in pipe.stream_infer(prompts,gen_config=gen):
                    row=rows[response.index]
                    if response.token_ids:
                        row['times'].append([time.monotonic()-began,len(response.token_ids)])
                        row['token_ids'].extend(response.token_ids)
                    row['finish_reason']=response.finish_reason
                ended=time.monotonic()
                assert all(len(r['token_ids'])==256 and r['finish_reason']=='length' for r in rows)
                workers=executor.collective_rpc('stage_collect')
                driver=stage_probe.take()
                data['events'].append(dict(batch=bs,rep=rep,began=began,ended=ended,
                    elapsed=ended-began,responses=rows,workers=workers,driver=driver))
                save()
                print('STAGE',bs,rep,round(ended-began,5),flush=True)
                if bs==1 and rep==0:
                    # Sampling starts after two initial batches and only affects diagnostic results.
                    data['stack_sampling'] = []
                    procs=[]
                    for name,proc in [('driver',data['driver']),('worker0',data['workers'][0]),('worker1',data['workers'][1])]:
                        cmd=[str(ROOT/'tools/bin/py-spy'),'record','--pid',str(proc['pid']),
                             '--duration','65','--rate','100','--idle','--nonblocking','--native',
                             '--format','speedscope','-o',str(ROOT/'results'/f'stage-{name}-stacks.json')]
                        log=(ROOT/'logs'/f'stage-{name}-pyspy.log').open('w')
                        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                        procs.append((process,log))
                        data['stack_sampling'].append(dict(command=cmd,start_monotonic=time.monotonic(),pid=process.pid))
        data['complete']=True
        save()
    for process,log in procs:
        process.wait(timeout=10);log.close()
    print('SAVED',out,flush=True)


if __name__=='__main__':
    main()
```
