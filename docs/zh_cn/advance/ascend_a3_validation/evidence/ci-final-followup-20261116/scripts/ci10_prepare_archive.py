"""Prepare immutable CI follow-up evidence and a GitHub blob upload manifest."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'artifacts/ci10-publication'
DOC = Path('docs/zh_cn/advance/ascend_a3_validation')
DEST = DOC / 'evidence/ci-final-followup-20261116'
HEAD = 'f64ab5061153b5c5b63721cf2d8d6f1b41b95104'


def git_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    state = json.loads(subprocess.check_output([
        'gh', 'api', f'repos/InternLM/lmdeploy/commits/{HEAD}/check-runs',
        '-H', 'Cache-Control: no-cache', '-f', 'per_page=100', '--method', 'GET']))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    snapshot = ROOT / 'inventory/ci10-checks-after-r0.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps({'time_utc': now, 'head': HEAD, **state}, indent=2) + '\n')
    summary = {
        'time_utc': now, 'head': HEAD,
        'first_attempt': {'passed': 6, 'failed': 2, 'cancelled': 5},
        'unit_test': {'passed': 6427, 'skipped': 400, 'failed': 0, 'warnings': 1284},
        'ete': {'conclusion': 'success', 'section_passed_counts': [20, 13, 10, 33],
                'automatic_reruns': 2, 'scope': 'Selected workflow tests; not six-model GPU certification'},
        'cuda313': {'status': 'hosted runner lost communication',
                    'last_log_utc': '2026-11-16T05:49:24.5545659Z',
                    'job_failed_utc': '2026-11-16T06:36:03Z',
                    'last_activity': 'TurboMind SM70 ptxas compilation',
                    'confirmed_oom': False, 'underlying_resource_or_network_cause': 'unresolved',
                    'rerun_job': 133270396913, 'rerun_attempt': 2,
                    'rerun_outcome_at_publication': 'in_progress'},
        'cancelled_matrix': 'Five annotations attribute cancellation to cu128/Python3.13 failure',
        'jetson': 'Inherited cuDSS1 SONAME versus Torch libcudss.so.0 requirement; unchanged recipe',
        'implementation_changes_this_round': False,
        'limitations': ['No runner kernel/OOM/network telemetry',
                       'No same-configuration CUDA3.13 baseline Docker rebuild on A3',
                       'Five cancelled jobs are not completed validations',
                       'One cancelled-job log download returned unexpected EOF twice; failure retained'],
    }
    summary_path = ROOT / 'results/ci10-summary-r0.json'
    assert not summary_path.exists()
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    sources = sorted((ROOT / 'inventory').glob('ci10-*.json'))
    sources += sorted((ROOT / 'logs').glob('ci10-job-*.log'))
    sources += [summary_path, Path(__file__)]
    files = []
    for source in sources:
        data = source.read_bytes()
        original_hash = hashlib.sha256(data).hexdigest()
        compressed = len(data) > 150_000
        name = source.relative_to(ROOT).as_posix() + ('.gz' if compressed else '')
        if compressed:
            data = gzip.compress(data, mtime=0)
        dest = PUB / DEST / name
        assert not dest.exists(), dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append({'path': name, 'source': str(source), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(), 'source_sha256': original_hash,
                      'compressed': compressed})
    index = {'head': HEAD, 'base': '442097e83799c2e85df9d6bbce35c7c79ea05dfe',
             'time_utc': now, 'files': files,
             'note': 'Raw failed retrieval responses retained alongside successful retries; they are not CI failures.'}
    (PUB / DEST / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    uploads = []
    for p in sorted(PUB.rglob('*')):
        if p.is_file():
            data = p.read_bytes()
            uploads.append({'path': p.relative_to(PUB).as_posix(), 'file': str(p),
                            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'git_blob_sha': git_sha(data)})
    manifest = ROOT / 'artifacts/ci10-upload-manifest-r0.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps(uploads, indent=2) + '\n')
    print(json.dumps({'evidence_files': len(files), 'upload_files': len(uploads),
                      'total_bytes': sum(x['bytes'] for x in uploads),
                      'max_blob_bytes': max(x['bytes'] for x in uploads)}, indent=2))


if __name__ == '__main__':
    main()
