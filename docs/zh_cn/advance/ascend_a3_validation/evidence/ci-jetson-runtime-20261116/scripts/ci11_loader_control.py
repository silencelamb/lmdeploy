"""Reproduce cuDSS loader-cache absence/presence in a fresh extracted package root.

This is the standalone form of the inline control saved as
results/ci11-cudss-loader-control-r0.json. It does not install a host package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

EXPECTED_SHA256 = 'f66d3099cbd962c6f6b135415e562d98cb5212c50a5d6286f0b860d10ecb50ac'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deb', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh output label'
    assert hashlib.sha256(args.deb.read_bytes()).hexdigest() == EXPECTED_SHA256
    host_cache = Path('/etc/ld.so.cache')
    original = hashlib.sha256(host_cache.read_bytes()).hexdigest()
    subprocess.run(['dpkg-deb', '-x', str(args.deb), str(args.output)], check=True)
    etc = args.output / 'etc'
    (etc / 'ld.so.conf.d').mkdir(parents=True)
    (etc / 'ld.so.conf').write_text('include /etc/ld.so.conf.d/*.conf\n')
    records = []
    for phase in ('before', 'after'):
        if phase == 'after':
            (etc / 'ld.so.conf.d/cudss.conf').write_text('/usr/lib/aarch64-linux-gnu/libcudss/12\n')
        command = ['ldconfig', '-r', str(args.output)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        query = ['ldconfig', '-p', '-C', str(etc / 'ld.so.cache')]
        cache = subprocess.check_output(query, text=True)
        present = 'libcudss.so.0 ' in cache
        assert present == (phase == 'after'), cache
        records.append({'phase': phase, 'command': command, 'query': query,
                        'present': present, 'cache': cache, 'stderr': result.stderr})
    assert hashlib.sha256(host_cache.read_bytes()).hexdigest() == original
    (args.output / 'control.json').write_text(json.dumps({
        'package_sha256': EXPECTED_SHA256, 'host_cache_unchanged': True,
        'scope': 'Loader-cache lookup only; no GPU inference', 'records': records}, indent=2) + '\n')
    print('Before: absent; after: SONAME0 resolves; host cache unchanged')


if __name__ == '__main__':
    main()
