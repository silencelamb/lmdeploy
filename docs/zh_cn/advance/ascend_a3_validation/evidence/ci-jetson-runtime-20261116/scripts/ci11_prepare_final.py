"""Prepare final 14-gate CI report, retaining all earlier indexed evidence."""
from pathlib import Path
import base64
import datetime
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ci10_publish_api import api as raw_api

HEAD = '2ccf38a43f0fdc7415ce4ce9f177852bdacc9966'
BRANCH = 'docs/ascend-a3-validation-20261114'
DOC = Path('docs/zh_cn/advance/ascend_a3_validation')
EVIDENCE = DOC / 'evidence/ci-jetson-runtime-20261116'
PUB = ROOT / 'artifacts/ci11-final-publication'


def api(path):
    for attempt in range(4):
        try:
            return raw_api(path)
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            time.sleep(1)


def write(path, text):
    target = PUB / path
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + '\n')


def main():
    assert not PUB.exists()
    audit = json.loads((ROOT / 'inventory/ci11-all-gates-passed-r0.json').read_text())
    assert audit['head'] == HEAD and audit['gate_count'] == 14
    assert all(c['conclusion'] == 'success' for c in audit['check_runs'])
    assert audit['commit_status']['state'] == 'success'
    assert audit['readthedocs']['success'] is True
    rtd_commands = json.loads((ROOT / 'inventory/ci11-rtd-rerun-commands-r0.json').read_text())
    assert rtd_commands['commit'] == HEAD and rtd_commands['success'] is True
    assert all(c['exit_code'] == 0 for c in rtd_commands['commands'])
    ete_log = (ROOT / 'logs/ci11-r1-job-133370945520-log-r0.txt').read_text(errors='replace')
    summary_lines = [line for line in ete_log.splitlines() if re.search(r'={3,}\s+\d+ passed\b', line)]
    passed = [int(re.search(r'(\d+) passed\b', line)[1]) for line in summary_lines]
    reruns = sum(int(m[1]) for line in summary_lines if (m := re.search(r'(\d+) rerun\b', line)))
    assert len(passed) == 17 and sum(passed) == 351 and reruns == 1
    summary = {'head': HEAD, 'gate_count': 14, 'all_actions_passed': 13,
               'all_legacy_statuses_passed': 1,
               'unit': {'passed': 7452, 'skipped': 400, 'warnings': 1903},
               'ete': {'pytest_phase_counts': passed, 'passed_executions': sum(passed),
                       'phases': len(passed), 'reruns': reruns,
                       'scope': 'Execution counts across model/backend phases, not unique test functions'},
               'readthedocs': {'failed_build': 35929337, 'successful_build': 35933214,
                              'same_commit': True, 'all_recorded_commands_exit_zero': True},
               'historical_summary_correction': 'ci10 ETE counts20/13/10/33 were a partial extraction, not the full workflow count',
               'audit': 'inventory/ci11-all-gates-passed-r0.json'}
    summary_path = ROOT / 'results/ci11-final-summary-r0.json'
    assert not summary_path.exists()
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    parent = api('git/ref/heads/' + BRANCH)['object']['sha']
    existing_paths = [DOC / 'README.md', DOC / 'mtp_ray_investigation_20261116.md',
                      DOC / 'ci_docker_recovery_20261116.md', EVIDENCE / 'index.json']
    originals = {str(path): api(f'contents/{path}?ref={parent}') for path in existing_paths}
    text = {path: base64.b64decode(item['content']).decode() for path, item in originals.items()}
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M Asia/Shanghai')
    report = text[str(DOC / 'ci_docker_recovery_20261116.md')]
    rows = ['| 检查 | 最终状态 | 原始任务 |', '| --- | --- | --- |']
    for c in audit['check_runs']:
        rows.append(f'| {c["name"]} | 通过 | [job{c["id"]}]({c["html_url"]}) |')
    rows.append('| docs/readthedocs.org:lmdeploy-zh-cn | 通过 | [build35933214](https://app.readthedocs.org/projects/lmdeploy-zh-cn/builds/35933214/) |')
    section = f'''## 远端CI结果

截至{stamp}，当前PR HEAD **`{HEAD}`的14/14项门禁全部成功**：13项GitHub Actions和1项Read the Docs commit status。
已同时核对PR当前HEAD、check-runs、commit statuses、`gh pr checks`及Read the Docs实际构建，不用旧HEAD或取消结果代替通过。

''' + '\n'.join(rows) + '''

八个CUDA/Python组合全部完成；Jetson和A3均完成原有构建及检查，没有删除测试或放宽断言。
完整unit为7452通过/400跳过；ETE有17段pytest结果，合计351次通过、1次自动rerun。
这些是不同模型/后端阶段的执行次数，不能当作351个互不重复的测试函数。

'''
    start = report.index('## 远端CI结果\n')
    end = report.index('### 当前单测计数与合并预览', start)
    report = report[:start] + section + report[end:]
    rtd_section = '''## Read the Docs的额外门禁与恢复

最终完整性审计发现，中文Read the Docs使用旧式commit status，而不在13项Actions的check-runs列表中。
初版13项清单不完整，已补入第14项；“13项Actions全绿”不能代表所有CI通过。

原构建35929337在checkout阶段失败：克隆和拉取`pull/5134/head`成功，但执行
`git checkout --force 2ccf38a43f0fdc7415ce4ce9f177852bdacc9966`报`unable to read tree`，尚未运行Sphinx。
随后实际远端PR引用已指向该提交。通过仓库既有、授权的中文RTD webhook重投递同一次
PR synchronize事件，保持代码SHA不变，也没有手工改写CI状态。

新构建35933214在同一提交上完成checkout、依赖安装和Sphinx HTML构建，所有记录命令exit code均为0，
Read the Docs实际success和GitHub对应status均变为成功。该对照符合瞬时引用可见性/同步时序问题；
原日志没有记录当时拉取引用的实际SHA，不能进一步断言服务内部的具体同步原因。

此前ci10的ETE摘要只摘取了带逗号/重跑标记的部分结果行（20/13/10/33），不是完整阶段计数。
原始日志和历史JSON保留；本节以完整解析后的17段、351次通过、1次rerun说明当前运行。

本轮通过的是上述CI门禁。400个skip没有计为硬件通过，Jetson实机推理和新增端到端性能测量仍不在覆盖范围。

'''
    report = report.replace('## 来源\n', rtd_section + '## 来源\n')
    write(DOC / 'ci_docker_recovery_20261116.md', report)
    readme = text[str(DOC / 'README.md')].splitlines()
    for i, line in enumerate(readme):
        if line.startswith('最新CI恢复：'):
            readme[i] = '最新CI恢复：PR HEAD为`2ccf38a4`，**14/14项门禁全部通过**（13项Actions及Read the Docs）。Jetson的cuDSS版本和库搜索路径问题已修复；8个CUDA组合、A3、lint、完整单测和ETE均完成。单测7452通过/400跳过，ETE按阶段合计351次通过。[CI修复与完整证据](ci_docker_recovery_20261116.md)保留失败迭代、RTD重建和验证范围。CANN继续保持9.1.0。'
    write(DOC / 'README.md', '\n'.join(readme))
    mtp = text[str(DOC / 'mtp_ray_investigation_20261116.md')].splitlines()
    for i, line in enumerate(mtp):
        if line.startswith('CI恢复接续：'):
            mtp[i] = 'CI恢复接续：当前PR `2ccf38a4` 的**14/14项CI门禁全部通过**，包含13项Actions及Read the Docs。Jetson版本/库路径修复、8个CUDA组合、完整unit/ETE及文档重建的证据见[独立CI恢复报告](ci_docker_recovery_20261116.md)。下文各轮状态保留为当时记录，旧ETE部分结果摘要的纠正见该报告。'
        if '日志分段为20/13/10/33项通过' in line:
            mtp[i] = line.replace('日志分段为20/13/10/33项通过，两个分段各含1次自动rerun', '旧摘要仅摘取带逗号或重跑标记的部分结果行，不能作为完整阶段计数；原始日志保留，当前完整解析见CI恢复报告')
    write(DOC / 'mtp_ray_investigation_20261116.md', '\n'.join(mtp))
    index = json.loads(text[str(EVIDENCE / 'index.json')])
    paths = [ROOT / 'inventory/ci11-all-gates-passed-r0.json', ROOT / 'inventory/ci11-legacy-status-r0.json',
             ROOT / 'inventory/ci11-rtd-redelivery-selection-r2.json', ROOT / 'inventory/ci11-rtd-redelivery-request-r0.json',
             ROOT / 'inventory/ci11-rtd-rerun-state-r0.json', ROOT / 'inventory/ci11-rtd-rerun-commands-r0.json',
             ROOT / 'inventory/ci11-final-log-retrieval-r1.json', ROOT / 'logs/ci11-final-pr-checks-r0.txt',
             ROOT / 'logs/ci11-r1-job-133370945520-log-r0.txt', ROOT / 'artifacts/ci11-rtd-api-r0.txt',
             ROOT / 'artifacts/ci11-rtd-api2-r0.txt', summary_path, Path(__file__)]
    paths += sorted((ROOT / 'logs').glob('ci11-final-*-r1.log'))
    paths += sorted((ROOT / 'inventory').glob('ci11-final-*-r1.json'))
    seen = {entry['path'] for entry in index['files']}
    for source in paths:
        raw = source.read_bytes()
        compressed = len(raw) > 150_000
        data = gzip.compress(raw, mtime=0) if compressed else raw
        name = source.relative_to(ROOT).as_posix() + ('.gz' if compressed else '')
        if name in seen:
            continue
        dest = PUB / EVIDENCE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        index['files'].append({'path': name, 'source': str(source), 'bytes': len(data),
                               'sha256': hashlib.sha256(data).hexdigest(),
                               'source_sha256': hashlib.sha256(raw).hexdigest(), 'compressed': compressed})
        seen.add(name)
    acceptance = (ROOT / 'ACCEPTANCE-ci11.md').read_bytes()
    acceptance_name = 'ACCEPTANCE-ci11-complete.md'
    dest = PUB / EVIDENCE / acceptance_name
    dest.write_bytes(acceptance)
    index['files'].append({'path': acceptance_name, 'source': str(ROOT / 'ACCEPTANCE-ci11.md'),
                           'bytes': len(acceptance), 'sha256': hashlib.sha256(acceptance).hexdigest(),
                           'source_sha256': hashlib.sha256(acceptance).hexdigest(), 'compressed': False})
    index.update(ci_complete=True, gate_count=14, final_audit='inventory/ci11-all-gates-passed-r0.json',
                 final_updated_utc=now.isoformat(), limits=['No Jetson GPU inference or new end-to-end performance measurement',
                                                          '400 skipped unit cases not counted as hardware passes'])
    write(EVIDENCE / 'index.json', json.dumps(index, indent=2))
    uploads = []
    for file in sorted(PUB.rglob('*')):
        if file.is_file():
            data = file.read_bytes()
            uploads.append({'path': file.relative_to(PUB).as_posix(), 'file': str(file), 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'git_blob_sha': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()})
    config = {'repository': 'silencelamb/lmdeploy', 'branch': BRANCH, 'initial_parent': parent,
              'expected_source_blobs': {path: obj['sha'] for path, obj in originals.items()}}
    for name, data in [('ci11-final-publish-config-r0.json', config), ('ci11-final-upload-manifest-r0.json', uploads)]:
        target = ROOT / 'artifacts' / name
        assert not target.exists()
        target.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'parent': parent, 'upload_files': len(uploads), 'evidence_files': len(index['files']),
                      'bytes': sum(x['bytes'] for x in uploads)}))


if __name__ == '__main__':
    main()
