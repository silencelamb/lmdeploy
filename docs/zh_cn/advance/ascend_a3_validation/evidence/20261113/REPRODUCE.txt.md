# REPRODUCE.txt

原始验证文件快照；源码 SHA256：`4110d8433051489fb0981467d030fd46b0680048aebdf8c42e371c9b954e9393`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```text
Follow-up for PR #5030, 2026-11-13 (Asia/Shanghai user-facing date).
Run timestamps remain UTC in command records for exact correlation.

cd /root/workspace/a3-mooncake-pr5030-20261113

Pinned sources:
- PR head: 1a09270b0835bb871be4eb4df70f9d04c8ccaec2 (open draft at inspection).
- baseline adapter: b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d, parent fb5de87a.
- native adapter: 4930080caac39c73427fcfd083d082fdebf7ec09, parent 6421cefe
  (contains the previously validated M1 import repair).
- baseline dlinfer remains 3b9096341a41e9d067104a39d914e47ca2d7ec50.
Both new branches contain the same PR adapter. The only merge resolution keeps
state_destination from the newer baseline. No remote push was performed.

The two venvs link to the preceding run's frozen dependencies, including NPU
Mooncake0.3.13.post1, but have their own editable project namespaces. Native has
no dlinfer and uses a mandatory import guard. setup_envs.py, install command
records, environment manifests and pip freeze preserve the construction.
The supplied image still fails pip check; this is not a clean-install certificate.

run.py imports the reviewed CANN/ATB environment exports and ACL_OP_INIT_MODE=1,
sets device visibility and logs argv, revisions, environment differences, timing,
exit code and a complete harness snapshot. Always use a new label/output.

SDK with actual bidirectional NPU data and the connector's scatter/gather API:
python scripts/run.py --env native --label replay-native-fabric --devices 2,3 \
  --timeout 240 -- env ASCEND_ENABLE_USE_FABRIC_MEM=1 \
  python scripts/fabric_probe.py --segment-gib 1 --io-api batch \
  --output results/replay-native-fabric.json
Repeat in baseline. The 8GiB control uses --segment-gib 8.
This owns a local master at55061/55062, so these ports and devices must be free.
The PR sets effective local_buffer_size=0, bypasses external-engine injection
and client HBM registration. Scalar get_into is a different API and fails with
no client staging buffer; its failure is preserved separately.

Model external cache test (two TP2 instances, four free logical devices0..3):
python scripts/run.py --env baseline --label replay-baseline-agents --devices 0,1,2,3 \
  --timeout 1500 -- env ASCEND_ENABLE_USE_FABRIC_MEM=1 \
  python scripts/model_external_cache.py --model /afs-weights/Agents-A1 --mtp 0 \
  --label replay-baseline-agents --output results/replay-baseline-agents.json
Repeat with --env native and a different label/output. Sage uses --model
/afs-weights/Sage-A1.5 --mtp 3. Both use segment8GiB, JSON local_buffer4GiB
(effective0), TP2/BF16/block128/prefill8192/session262144/cache0.95/prefix state
budget16/alignment128/last-only. Fixed seeds and no-thinking apply to generation.

The producer on devices0,1 remains alive after four distinct long requests,
retaining its Store segments. A fresh consumer on2,3 first runs an independent
cold warmup, then the identical four requests. This prevents local prefix hits
from masquerading as external-cache hits. One shared request is warmup; three
are measured. Requests contain distinct verification codes to test state reuse.
Worker logs establish actual puts, gets and GDN state transfer on both ranks.

python scripts/collect_runtime_logs.py
python scripts/analyze_external_cache.py results/replay-baseline-agents.json \
  results/replay-native-agents.json --output results/replay-agents-comparison.json
The analyzer asserts same model/settings/prompts, compares common-history scores,
checks tokens and records cold/external TTFT. Timing is a small C1 sample and does
not replace the previous HTTP C4/C16 measurements or production load testing.

local_buffer_control.py isolates the earlier setup issue: an aligned1GiB global
segment and the same runtime, with effective --local-mib 0 versus256. This tests
initialization only and is not counted as a data-transfer acceptance test.

Original 2026-11-12 logs and evidence.tar.gz remain unchanged in
/root/workspace/a3-validation-20261112. The follow-up conclusions supersede only
the prior Mooncake blocker interpretation, not the previous MTP/HTTP findings.

TileLang follow-up (same date; completed in both active environments):
- tilelang==0.1.15
- apache-tvm-ffi==0.1.12 (replaces these venvs' link to 0.1.14.post1)
- torch-c-dlpack-ext==0.1.5
- ml-dtypes==0.6.0
- z3-solver==4.15.4.0
Wheel URLs, filenames, hashes and sizes are pinned in
inventory/tilelang-followup/selected-wheels.json; verified wheels are retained
under artifacts/tilelang-wheels and included in the updated evidence archive.
scripts/download_tilelang_wheels.py obtains the exact files with hash checks.
For a fresh reconstruction, run scripts/setup_envs.py first, then install once:
python scripts/run.py --env native --label replay-install-tilelang --devices 5 \
  --timeout 300 -- python scripts/install_tilelang.py
This installs only the five verified wheels with --no-index --no-deps into
envs/tilelang-0.1.15-dependencies and links them into both isolated environments.
It deliberately refuses to overwrite an existing bundle; the current workspace
is already installed. Prior frozen dependencies and the global image are intact.
The environment-changes.json plus baseline/native freeze files under
inventory/tilelang-followup capture the exact dependency change and final state.

Replay the complete worker suites (use unique labels if repeated):
python scripts/run.py --env baseline --label replay-baseline-worker-tilelang \
  --devices 4 --timeout 300 -- python -m pytest -q \
  sources/lmdeploy-baseline/tests/pytorch/kv_connector/test_mooncake_store_worker.py \
  --junitxml=results/replay-baseline-worker-tilelang.xml
python scripts/run.py --env native --label replay-native-worker-tilelang \
  --devices 5 --timeout 300 -- python -m pytest -q \
  sources/lmdeploy-native/tests/pytorch/kv_connector/test_mooncake_store_worker.py \
  --junitxml=results/replay-native-worker-tilelang.xml
Both measured suites now pass 101/101, including the previously failing GLM MTP
cache roundtrip. The initial 100-pass/1-fail XML and logs remain available.
scripts/audit_tilelang.py checks pinned versions, declared requirements, real
TileLang imports, XGrammar parsing, no additional pip-check errors and native
dlinfer isolation. Run through run.py separately in each environment.
Existing image-wide pip-check failures remain; GLM full-model/CUDA kernel tests
were not added. Previous NPU model results predate this dependency installation.
The earlier follow-up archive is preserved as evidence-before-tilelang.tar.gz;
evidence.tar.gz now includes this dependency installation and retest evidence.

```
