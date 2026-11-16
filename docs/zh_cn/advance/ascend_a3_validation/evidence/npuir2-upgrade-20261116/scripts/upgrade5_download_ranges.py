"""Range download official artifacts; validate sizes and hash assembled files."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/upgrade5-downloads-r1'
DEST.mkdir(exist_ok=False)
URLS = [
    'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.0/Ascend-cann-toolkit_9.2.0_linux-aarch64.run',
    'https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.2.post2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl',
]

def download(url):
    path = DEST / url.rsplit('/', 1)[1]
    record = dict(url=url, path=str(path), start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'Referer':'https://www.hiascend.com/'})
        with urllib.request.urlopen(req, timeout=45) as response:
            record['headers'] = dict(response.headers)
            size = int(response.headers['Content-Length'])
        def part(start):
            end = min(size, start + 32*1024*1024)-1
            req = urllib.request.Request(url, headers={'Referer':'https://www.hiascend.com/', 'Range':f'bytes={start}-{end}'})
            p = path.with_name(path.name + f'.part{start:012d}')
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120) as response, p.open('wb') as output:
                        assert response.status == 206, (response.status,start)
                        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}', response.headers
                        while chunk := response.read(1024*1024): output.write(chunk)
                    assert p.stat().st_size == end-start+1
                    return p
                except Exception:
                    if attempt == 2: raise
                    time.sleep(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            parts = list(pool.map(part, range(0,size,32*1024*1024)))
        sha = hashlib.sha256(); md5 = hashlib.md5()
        with path.open('xb') as output:
            for p in parts:
                with p.open('rb') as inp:
                    while chunk := inp.read(8*1024*1024):
                        output.write(chunk);sha.update(chunk);md5.update(chunk)
        record.update(status='downloaded', size=path.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())
        assert record['size'] == size
    except Exception as e:
        record.update(status='failed', error=repr(e))
    record['finish'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.with_name(path.name + '.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(download, URLS))
(ROOT/'inventory/upgrade5-downloads-r1.json').write_text(json.dumps(results,indent=2)+'\n')
