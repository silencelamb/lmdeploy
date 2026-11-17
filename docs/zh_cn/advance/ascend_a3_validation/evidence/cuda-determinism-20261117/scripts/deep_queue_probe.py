"""CPU control of the actual RequestManager drain rule, not a model benchmark."""
import asyncio
import json
from pathlib import Path
import time
from lmdeploy.pytorch.engine.request import RequestManager,Request,RequestType

ROOT=Path(__file__).resolve().parents[1]
async def trial(delayed):
    mgr=RequestManager();mgr.requests=asyncio.Queue();origin=time.perf_counter_ns();events=[]
    def put(i):
        mgr.requests.put_nowait(Request(type=RequestType.ADD_MESSAGE,sender_id=i,data={'id':i},resp=None))
        events.append(dict(kind='enqueue',id=i,ns=time.perf_counter_ns()-origin))
    for i in range(3):put(i)
    async def late():await asyncio.sleep(.001);put(3)
    if delayed:task=asyncio.create_task(late())
    else:put(3);task=None
    batches=[]
    while sum(len(x) for x in batches)<4:
        requests=await mgr.get_all_requests();group=[x.data['id'] for x in requests[RequestType.ADD_MESSAGE]]
        batches.append(group);events.append(dict(kind='drain',ids=group,ns=time.perf_counter_ns()-origin))
    if task:await task
    return dict(late_fourth=delayed,batches=batches,events=events)
async def main():
    rows=[await trial(delay) for delay in [False,True,False,True]]
    path=ROOT/'followup-20261117/queue-drain.json';path.write_text(json.dumps(dict(scope='Actual RequestManager.get_all_requests with synthetic arrival timing; excludes GPU/HTTP performance claims.',trials=rows),indent=2))
    print([(x['late_fourth'],x['batches']) for x in rows])
if __name__=='__main__':asyncio.run(main())
