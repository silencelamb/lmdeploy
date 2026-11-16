"""Download official compiler inputs without installing into any environment."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/upgrade5-downloads'
DEST.mkdir(exist_ok=False)
URLS = [
    'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.0/Ascend-cann-toolkit_9.2.0_linux-aarch64.run',
    'https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.2.post2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl',
]


def download(url):
    path = DEST / url.rsplit('/', 1)[1]
    record = dict(url=url, path=str(path), start=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        req = urllib.request.Request(url, headers={'Referer': 'https://www.hiascend.com/'})
        digest = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=120) as response, path.open('xb') as f:
            record['headers'] = dict(response.headers)
            while chunk := response.read(8 * 1024 * 1024):
                f.write(chunk)
                digest.update(chunk)
        record.update(size=path.stat().st_size, sha256=digest.hexdigest(), status='downloaded')
        assert record['size'] == int(record['headers'].get('Content-Length', record['size']))
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
    record['finish'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.with_name(path.name + '.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(download, URLS))
(ROOT / 'inventory/upgrade5-downloads-r0.json').write_text(json.dumps(results, indent=2) + '\n')
