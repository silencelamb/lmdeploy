"""Fetch completed GitHub job logs with validated HTTP ranges and bounded retries."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def request(job, start, end):
    command = ['gh', 'api', f'repos/InternLM/lmdeploy/actions/jobs/{job}/logs',
               '-H', f'Range: bytes={start}-{end}', '-H', 'Accept-Encoding: identity', '--include']
    result = subprocess.run(command, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    header, sep, body = result.stdout.partition(b'\r\n\r\n')
    if not sep:
        header, sep, body = result.stdout.partition(b'\n\n')
    assert sep and b'206 Partial Content' in header, header[:200]
    content_range = re.search(rb'(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)', header)
    assert content_range
    actual_start, actual_end, size = map(int, content_range.groups())
    assert actual_start == start and actual_end == min(end, size - 1)
    assert len(body) == actual_end - start + 1
    etag = re.search(rb'(?im)^etag:\s*([^\r\n]+)', header).group(1).decode()
    return body, size, etag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.label)
    out = ROOT / 'logs' / f'{args.label}.log'
    metadata = ROOT / 'inventory' / f'{args.label}.json'
    assert not out.exists() and not metadata.exists()
    errors = []
    for attempt in range(4):
        try:
            _, size, etag = request(args.job, 0, 1023)
            break
        except Exception as error:
            errors.append({'probe_attempt': attempt, 'error': repr(error)})
            if attempt == 3:
                raise
            time.sleep(1)
    chunk = 256 * 1024
    specs = [(start, min(size - 1, start + chunk - 1)) for start in range(0, size, chunk)]

    def fetch(spec):
        start, end = spec
        attempts = []
        for attempt in range(4):
            try:
                data, actual_size, actual_etag = request(args.job, start, end)
                assert actual_size == size and actual_etag == etag
                return data, {'start': start, 'end': end, 'retry_errors': attempts}
            except Exception as error:
                attempts.append({'attempt': attempt, 'error': repr(error)})
                if attempt == 3:
                    raise
                time.sleep(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pieces = list(pool.map(fetch, specs))
    data = b''.join(x[0] for x in pieces)
    assert len(data) == size
    out.write_bytes(data)
    record = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'job_id': args.job, 'bytes': size, 'etag': etag,
              'sha256': hashlib.sha256(data).hexdigest(), 'probe_errors': errors,
              'ranges': [x[1] for x in pieces], 'script': __file__}
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('ranges', 'probe_errors')}))


if __name__ == '__main__':
    main()
