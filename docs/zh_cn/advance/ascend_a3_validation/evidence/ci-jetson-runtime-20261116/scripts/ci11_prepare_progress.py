"""Freeze current CI evidence and prepare documentation on the latest remote tree."""
from pathlib import Path
import base64
import datetime
import gzip
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ci10_publish_api import api as raw_api

HEAD = '2ccf38a43f0fdc7415ce4ce9f177852bdacc9966'
BRANCH = 'docs/ascend-a3-validation-20261114'
DOC = Path('docs/zh_cn/advance/ascend_a3_validation')
PUB = ROOT / 'artifacts/ci11-progress-publication'
EVIDENCE = DOC / 'evidence/ci-jetson-runtime-20261116'


def api(path):
    for attempt in range(4):
        try:
            return raw_api(path)
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            time.sleep(1)


def put(path, content):
    target = PUB / path
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + '\n')


def main():
    assert not PUB.exists()
    parent = api('git/ref/heads/' + BRANCH)['object']['sha']
    originals = {}
    for name in ['README.md', 'mtp_ray_investigation_20261116.md']:
        path = str(DOC / name)
        obj = api(f'contents/{path}?ref={parent}')
        originals[path] = obj
    for attempt in range(4):
        result = subprocess.run(['gh', 'api', f'repos/InternLM/lmdeploy/commits/{HEAD}/check-runs',
                                 '-H', 'Cache-Control: no-cache', '-f', 'per_page=100', '--method', 'GET'],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            break
        time.sleep(1)
    assert result.returncode == 0, result.stderr
    checks = json.loads(result.stdout)['check_runs']
    now = datetime.datetime.now(datetime.timezone.utc)
    local = now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M Asia/Shanghai')
    snapshot = ROOT / 'inventory/ci11-progress-checks-r0.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps({'time_utc': now.isoformat(), 'head': HEAD, 'checks': checks}, indent=2) + '\n')
    draft = (ROOT / 'artifacts/ci11-report-draft.md').read_text()
    status = {'success': '通过', 'failure': '失败', 'cancelled': '取消',
              'in_progress': '执行中', 'queued': '排队'}
    rows = ['| 检查 | 当前状态 | 原始任务 |', '| --- | --- | --- |']
    for c in checks:
        state = status.get(c['conclusion'] or c['status'], c['conclusion'] or c['status'])
        rows.append(f'| {c["name"]} | {state} | [job{c["id"]}]({c["html_url"]}) |')
    old = '待当前HEAD全部检查完成后，由原始check-run、job步骤和日志填写；当前不能计为全部通过。'
    draft = draft.replace(old, f'截至{local}的当前HEAD结果如下。仍在执行的检查未计为通过，持续跟进目标保持进行中。\n\n' + '\n'.join(rows))
    draft = draft.replace('完整远端结果收齐后再更新最终表格；\n不能把进行中、取消或旧HEAD的通过状态计为本轮通过。',
                          '本文按实际完成结果更新；\n不能把进行中、取消或旧HEAD的通过状态计为本轮通过。')
    insertion = '''
### 当前单测计数与合并预览

当前完整单测为**7452通过、400跳过、1903条warning**。本轮CI实际checkout的是
`fea0d6fd9975d1a23d260010eef4b10d95ed5571`，父提交为当前上游
`95d2d6280ed4946fd498d6026f338c7bd41bfb63`和PR HEAD `2ccf38a4`。
上轮6427通过的CI checkout为`0e15234989953acd0fa9dbae4573545ab2b6c6cf`，父提交是
`442097e8`和`f64ab506`。新增的1025个用例来自期间合入上游的Anthropic、XML/schema
和GLM聊天模板测试；它们不应算作这三个CI修复提交新增的测试。硬件、Python3.12.13、
Torch2.10.0+cu128和TorchVision0.25.0+cu128保持一致。

固定版本的A3迁移比较与这个GitHub合并预览测试是不同的验证记录，不能把旧模型结果
重标为新上游版本上的测量。单测400项skip没有计入通过数。

Jetson当前已经实际通过Build Docker image、`lmdeploy check_env`和Dive三步。
这确认了版本选择与加载路径修复；实机GPU推理仍未在本轮覆盖。

'''
    draft = draft.replace('## 来源\n', insertion + '## 来源\n')
    draft = draft.replace('## 本地验证与证据\n', '## 本地验证与证据\n\n[完整证据索引](evidence/ci-jetson-runtime-20261116/index.json)记录原始日志、命令、失败、包哈希与对照。\n')
    put(DOC / 'ci_docker_recovery_20261116.md', draft)
    readme = base64.b64decode(originals[str(DOC / 'README.md')]['content']).decode()
    lines = readme.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('最新交付：PR HEAD为'):
            lines[i] = '最新CI恢复：PR HEAD为`2ccf38a4`。Jetson的cuDSS版本和库搜索路径问题已修复，当前Jetson、A3、lint及完整单测已通过；单测7452通过/400跳过，包含合并预览中的上游新增测试。CUDA矩阵和ETE仍在执行，未计为通过。[CI修复与当前结果](ci_docker_recovery_20261116.md)保存独立归因与证据。CANN继续保持9.1.0。'
        if '[LMDeploy `pr/ascend-a3-native`]' in line:
            lines[i] = line.replace('f64ab5061153b5c5b63721cf2d8d6f1b41b95104', HEAD).replace('`f64ab506`', '`2ccf38a4`')
    put(DOC / 'README.md', '\n'.join(lines))
    report = base64.b64decode(originals[str(DOC / 'mtp_ray_investigation_20261116.md')]['content']).decode()
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('CI接续：'):
            lines[i] = 'CI恢复接续：当前PR为`2ccf38a4`。Jetson库版本与搜索路径问题已修复并通过真实镜像检查；当前完整单测7452通过/400跳过。其余检查继续跟进，详见[独立CI恢复报告](ci_docker_recovery_20261116.md)。下文各轮CI结果保留为当时状态。'
    put(DOC / 'mtp_ray_investigation_20261116.md', '\n'.join(lines))
    source_names = [
        'ACCEPTANCE-ci11.md', 'inventory/ci11-checks-before-r0.json',
        'inventory/ci11-cudss-apt-r0.json', 'inventory/ci11-cudss-elf-r1.json',
        'inventory/ci11-cudss-package-layout-r0.json', 'inventory/ci11-cudss-download-failure-r0.json',
        'inventory/ci11-final-source-r0.json', 'inventory/ci11-final-source-r1.json',
        'inventory/ci11-jetson-passed-r0.json', 'inventory/ci11-jetson-passed-log-r0.json',
        'inventory/ci11-jetson-failure-r1.json', 'inventory/ci11-unit-passed-log-r0.json',
        'inventory/ci11-ci-checkout-audit-r0.json', 'inventory/ci11-upstream-test-delta-r0.json',
        'inventory/ci11-progress-checks-r0.json', 'inventory/ci11-unit-ete-progress-r0.json',
        'results/ci11-cudss-loader-control-r0.json',
        'logs/ci11-jetson-failure-r1.log', 'logs/ci11-jetson-passed-log-r0.log',
        'logs/ci11-unit-passed-log-r0.log', 'logs/ci11-cudss-download-r1.log',
        'logs/ci11-precommit-r0.log', 'logs/ci11-precommit-r0.json',
        'logs/ci11-precommit-r1.log', 'logs/ci11-precommit-r1.json',
        'logs/ci11-push-r0.log', 'logs/ci11-push-r0.json',
        'logs/ci11-push-r1.log', 'logs/ci11-push-r1.json',
        'logs/ci11-loader-repro-r1.log', 'logs/ci11-loader-repro-r1.json',
        'scripts/ci11_cudss_download.py', 'scripts/ci11_loader_control.py',
        'scripts/ci_fetch_job_log.py', 'scripts/ci11_prepare_progress.py',
        'artifacts/ci11-loader-repro-r1/control.json',
    ]
    index = []
    for name in source_names:
        source = ROOT / name
        raw = source.read_bytes()
        compressed = len(raw) > 150_000
        data = gzip.compress(raw, mtime=0) if compressed else raw
        relative = name + ('.gz' if compressed else '')
        dest = PUB / EVIDENCE / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        index.append({'path': relative, 'source': str(source), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'source_sha256': hashlib.sha256(raw).hexdigest(), 'compressed': compressed})
    put(EVIDENCE / 'index.json', json.dumps({'head': HEAD, 'time_utc': now.isoformat(),
        'ci_complete': False, 'files': index, 'limits': ['CUDA/ETE still executing at this snapshot',
        'No Jetson GPU inference or new end-to-end performance measurement']}, indent=2))
    files = []
    for file in sorted(PUB.rglob('*')):
        if file.is_file():
            data = file.read_bytes()
            files.append({'path': file.relative_to(PUB).as_posix(), 'file': str(file),
                          'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                          'git_blob_sha': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()})
    config = {'repository': 'silencelamb/lmdeploy', 'branch': BRANCH, 'initial_parent': parent,
              'expected_source_blobs': {path: item['sha'] for path, item in originals.items()}}
    for name, obj in [('ci11-progress-publish-config-r0.json', config),
                      ('ci11-progress-upload-manifest-r0.json', files)]:
        out = ROOT / 'artifacts' / name
        assert not out.exists()
        out.write_text(json.dumps(obj, indent=2) + '\n')
    print(json.dumps({'parent': parent, 'files': len(files), 'evidence_files': len(index),
                      'total_bytes': sum(x['bytes'] for x in files)}))


if __name__ == '__main__':
    main()
