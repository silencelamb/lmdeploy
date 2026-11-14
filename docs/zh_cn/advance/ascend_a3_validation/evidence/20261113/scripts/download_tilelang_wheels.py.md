# download_tilelang_wheels.py

原始验证文件快照；源码 SHA256：`c14d8b22a7a58533057f8121a1adda4bea30e67757ba5e904d8f79b0bf9595b2`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Download pinned public wheels in ranges and verify PyPI SHA256 before use."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
selected=json.loads((ROOT/'inventory/tilelang-followup/selected-wheels.json').read_text())
dest=ROOT/'artifacts/tilelang-wheels';dest.mkdir(exist_ok=True)
jobs=[];fds={};started=time.monotonic()
for item in selected:
    path=dest/item['filename']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:continue
    tmp=path.with_suffix('.whl.download')
    fd=os.open(tmp,os.O_CREAT|os.O_RDWR|os.O_TRUNC,0o644);os.ftruncate(fd,item['size']);fds[item['filename']]=fd
    for start in range(0,item['size'],1024**2):
        jobs.append((item,start,min(item['size']-1,start+1024**2-1),fd))

def fetch(job):
    item,start,end,fd=job
    for attempt in range(3):
        try:
            req=urllib.request.Request(item['url'],headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=45) as response:
                assert response.status==206,response.status
                assert response.headers['Content-Range']==f"bytes {start}-{end}/{item['size']}"
                data=response.read()
            assert len(data)==end-start+1
            assert os.pwrite(fd,data,start)==len(data)
            return item['filename'],start,end
        except Exception:
            if attempt==2:raise
            time.sleep(1)

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(fetch,job) for job in jobs]
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            filename,start,end=future.result()
            print(n,'/',len(jobs),filename,start,end,flush=True)
finally:
    for fd in fds.values():os.close(fd)
for item in selected:
    path=dest/item['filename'];tmp=path.with_suffix('.whl.download')
    if tmp.exists():
        assert hashlib.sha256(tmp.read_bytes()).hexdigest()==item['sha256'],item['filename']
        tmp.rename(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
print('Verified all wheels in',time.monotonic()-started,'seconds',flush=True)

```
