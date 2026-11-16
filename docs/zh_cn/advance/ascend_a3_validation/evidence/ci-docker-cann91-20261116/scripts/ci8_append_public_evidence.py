"""Append the completed CANN9.1 Docker and targeted CI evidence, without replacing logs."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path('/root/workspace/a3-publish-20261114/lmdeploy-reports/docs/zh_cn/advance/'
            'ascend_a3_validation/evidence/ci-docker-cann91-20261116')
FILES = [
    'inventory/ci8-checks-after-r2.json',
    'inventory/ci8-final-state-r2.json',
    'inventory/ci8-cann91-image-config-r0.json',
    'logs/ci8-ascend-docker-passed-r0.log',
    'logs/ci8-unit-cancelled111-r0.log',
    'logs/ci8-jetson-final-r0.log',
    'logs/ci8-clean-runtime-check-r0.log',
    'logs/ci8-clean-runtime-check-r0.json',
    'results/ci8-a100-partial111-r0.json',
    'results/ci8-summary-r2.json',
    'scripts/ci8_append_public_evidence.py',
]


def main():
    index_path = DEST / 'index.json'
    index = json.loads(index_path.read_text())
    existing = {entry['path']: entry for entry in index['files']}
    for name in FILES:
        source = ROOT / name
        original = source.read_bytes()
        compressed = len(original) > 750_000
        target = name + '.gz' if compressed else name
        data = gzip.compress(original, mtime=0) if compressed else original
        digest = hashlib.sha256(data).hexdigest()
        if target in existing:
            assert existing[target]['sha256'] == digest, target
            continue
        destination = DEST / target
        assert not destination.exists(), destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        index['files'].append(dict(path=target, source=str(source), bytes=len(data),
                                   sha256=digest, source_sha256=hashlib.sha256(original).hexdigest(),
                                   compressed=compressed))
    index['ci_state'] = 'f64 A3 Docker and lint passed; Jetson inherited ABI failure; full A100/CUDA/ETE pending'
    index['latest_status_record'] = 'inventory/ci8-final-state-r2.json'
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    for entry in index['files']:
        data = (DEST / entry['path']).read_bytes()
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    print('Verified evidence files:', len(index['files']))


if __name__ == '__main__':
    main()
