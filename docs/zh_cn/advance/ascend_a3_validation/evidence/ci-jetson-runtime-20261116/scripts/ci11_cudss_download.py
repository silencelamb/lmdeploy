"""Download the verified runtime in bounded ranges; inspect ELF without installing it."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcudss0-cuda-12_0.8.0.10-1_arm64.deb'
SIZE = 36461572
SHA256 = 'f66d3099cbd962c6f6b135415e562d98cb5212c50a5d6286f0b860d10ecb50ac'
PARTS = ROOT / 'artifacts/ci11-cudss-parts-r1'
CHUNK = 4 * 1024 * 1024


def part(spec):
    start, end = spec
    path = PARTS / f'{start:09d}-{end:09d}.part'
    assert not path.exists()
    errors = []
    for attempt in range(4):
        offset = start + (path.stat().st_size if path.exists() else 0)
        if offset == end + 1:
            break
        try:
            req = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{end}'})
            with urllib.request.urlopen(req, timeout=60) as response:
                assert response.status == 206, response.status
                assert response.headers['Content-Range'] == f'bytes {offset}-{end}/{SIZE}'
                deadline = time.monotonic() + 180
                with path.open('ab') as out:
                    while data := response.read(64 * 1024):
                        out.write(data)
                        if time.monotonic() > deadline:
                            raise TimeoutError('range exceeded180s; resume partial bytes')
            assert path.stat().st_size == end - start + 1
            break
        except Exception as error:
            errors.append({'attempt': attempt, 'offset': offset, 'error': repr(error)})
            time.sleep(1)
    assert path.stat().st_size == end - start + 1, (path, errors)
    return {'start': start, 'end': end, 'bytes': path.stat().st_size, 'errors': errors}


def main():
    assert not PARTS.exists()
    PARTS.mkdir()
    specs = [(start, min(SIZE - 1, start + CHUNK - 1)) for start in range(0, SIZE, CHUNK)]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(part, specs):
            records.append(result)
            print(json.dumps(result), flush=True)
    data = b''.join((PARTS / f'{start:09d}-{end:09d}.part').read_bytes() for start, end in specs)
    assert len(data) == SIZE and hashlib.sha256(data).hexdigest() == SHA256
    deb = ROOT / 'artifacts/ci11-libcudss0-0.8.0.10-r1.deb'
    assert not deb.exists()
    deb.write_bytes(data)
    directory = ROOT / 'artifacts/ci11-cudss-runtime-r1'
    assert not directory.exists()
    subprocess.run(['dpkg-deb', '-x', str(deb), str(directory)], check=True)
    so = next(p for p in directory.rglob('libcudss.so.*') if p.is_file() and not p.is_symlink())
    dynamic = subprocess.check_output(['readelf', '-d', str(so)], text=True)
    assert '(SONAME)' in dynamic and '[libcudss.so.0]' in dynamic
    out = ROOT / 'inventory/ci11-cudss-elf-r1.json'
    assert not out.exists()
    out.write_text(json.dumps({'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'url': URL, 'bytes': SIZE, 'sha256': SHA256, 'ranges': records,
                              'library': str(so.relative_to(directory)), 'elf_dynamic': dynamic}, indent=2) + '\n')
    print('Verified official package SHA256 and SONAME libcudss.so.0', flush=True)


if __name__ == '__main__':
    main()
