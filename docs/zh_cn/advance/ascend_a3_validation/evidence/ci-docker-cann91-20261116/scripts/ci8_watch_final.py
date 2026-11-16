"""Persist fresh GitHub check-run state and download failed completed job logs."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
HEAD='f64ab5061153b5c5b63721cf2d8d6f1b41b95104'
previous=None
seen_failures=set()
start=time.monotonic()
iteration=0
while time.monotonic()-start<7200:
    result=subprocess.run(['gh','api',f'repos/InternLM/lmdeploy/commits/{HEAD}/check-runs',
                           '-H','Cache-Control: no-cache'],capture_output=True,text=True)
    record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode)
    if result.returncode:
        record['error']=result.stderr
        (ROOT/'inventory'/f'ci8-final-watch-{iteration:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    else:
        d=json.loads(result.stdout)
        record['checks']=[dict(id=x['id'],name=x['name'],status=x['status'],conclusion=x['conclusion'],url=x['html_url']) for x in d['check_runs']]
        (ROOT/'inventory'/f'ci8-final-watch-{iteration:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
        key=tuple(sorted((x['id'],x['status'],x['conclusion']) for x in record['checks']))
        if key!=previous:
            print(json.dumps(record),flush=True);previous=key
        for check in record['checks']:
            if check['status']=='completed' and check['conclusion']=='failure' and check['id'] not in seen_failures:
                out=ROOT/'logs'/f'ci8-new-job-{check["id"]}.log'
                if not out.exists():
                    with out.open('x') as stream:
                        subprocess.run(['gh','api',f'repos/InternLM/lmdeploy/actions/jobs/{check["id"]}/logs'],stdout=stream,stderr=subprocess.STDOUT)
                seen_failures.add(check['id'])
        if record['checks'] and all(x['status']=='completed' for x in record['checks']):
            print('ALL_CHECKS_COMPLETED',flush=True);break
    iteration+=1
    time.sleep(45)
