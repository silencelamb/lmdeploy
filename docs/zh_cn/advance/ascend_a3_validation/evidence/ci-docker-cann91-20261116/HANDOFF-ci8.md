# Active: CANN9.1 Docker/dependencies, CI5134 fixes, GPU handoff

Latest user: update Dockerfile and dependencies to CANN9.1.0 (do NOT resume9.2),
audit/fix PR5134 CI regressions vs baseline, explain remaining Ray work, provide
a GPU validation prompt. Original active environment must stay unchanged.

PR remote head is now111dbf6b4dc1c1aa9fb2fd0753313e0314858c96 (pushed), with
sampling fix143fa1e647b4029687c5fc10bdc54d16620d4d13 before it; base442097e8.
Independent checkout `lmdeploy-ci8`, branch `research/ascend-cann91-ci-20261116`,
is clean and the formal PR worktree was fast-forwarded too. Reports branch still
2cc92dd8 at `/root/workspace/a3-publish-20261114/lmdeploy-reports` pending ci8 docs.

CI evidence saved as logs/ci8-*-github-r0.log and inventory/ci8-checks-before-r0.json.
Current85 jobs: lint133073239673 fails2 dead links; A3 Docker133073239872 fails
GitCode clone418; Jetson133073240108 builds but importTorch fails libcudss.so.0.
Unit133073239883/run43798653185 still running; API job log returns404 until done.
Previous completed unit job132851013103 has3 cache_config AttributeErrors.

IMPORTANT attribution correction: those3 unit failures ARE introduced by the PR.
Same CPU env: base442097e8 passes3/3, PR85 fails3/3. Files:
logs/ci8-base-sampling-r0.*, ci8-pr-sampling-before-r0.*. Fix in agent.py checks
the actual tensor's NPU device, optional target_logits for MTP, instead of
requiring self.cache_config in sampling. Entire test_model_agent.py now175pass/
7skip (CPU hardware limits), log ci8-pr-sampling-fixed-r0.

Jetson inherited root: unchanged docker/Dockerfile.jetson and CUDA requirements
install unpinned cudss, which now resolves to libcudss1-cuda-12 1.0.0.35, while
Jetson Torch needs SONAME0. Log records both. No Jetson fix mixed into native PR;
complete separate base-image rerun not done. State source invariance explicitly.

Dependency fix prepared:
- Docker official quay.io/ascend/cann:9.1.0-a3-openeuler24.03-py3.12; tag verified
  via Quay API, manifest recorded. S WR HTTP401 was ordinary registry auth probe,
  not evidence that image is absent.
- Own application venv; CPU Torch2.10.0+cpu, TorchNPU2.10.0.post4,
  torchvision0.25.0+cpu, Triton-Ascend3.2.2.post2, Ray2.48.0,
  transformers5.14.1, NumPy1.26.4 (Triton declared bound) in new constraints file.
- Widen native-only torch bounds to<2.11 and torchvision<0.26; CUDA unchanged.
- Docker predownloads CPU wheels and Ascend Triton from respective official
  indexes, then resolves whole app from PyPI with local find-links/constraints.
  Runs pip check and import/no-dlinfer/no-external-kernel check after install.
- GitCode external kernels master also hard-pins incompatible Torch2.7.1/
  Triton3.2.0. No official PyPI package or GitHub mirror found. Moved ONLY the
  actually used 11-file/1670-line GDN prefill+L2Norm slice from the previously
  validated frozen8e70 snapshot into native fla/prefill; retain MIT LICENSE,
  per-file notices/provenance and existing cumsum destination-dtype + CANN slice
  compatibility edits. Remove floating external dependency. No arithmetic change.
- Dead links replaced with verified pinned dlinfer migration source3b909 and
  FLA0.5.2 source commit9c8e42e. Bilingual guides and kernel README updated.

Completed checks:
- All11 executable ASTs equal to frozen reference after excluding imports and
  cosmetic type/docstring changes; preserve tl.constexpr annotations in comparison.
- L2Norm6 + prefill6 (dense/varlen, with/without initial state) bitwise exact,
  including final states: results/ci8-prefill-equivalence-r0.json.
- Six circular hardware cases pass with external-kernel imports blocked.
- Sage MTP2 fresh trace21 runs and Agents before/after trace21 each match every
  token and score bit: results/ci8-model-comparison-r0.json.
- Applicable precommit r1 passes; interrogate70.3% and pylint9.23/10 pass;
  module-init and README doc link scripts pass. All current CI commands are logged.
- Wheel/sdist build completed; wheel includes11 prefills plusMIT LICENSE and no
  dlinfer/external-kernel dependency. The combined package+NumPy verifier was
  invoked before its model run completed, so it stopped at that assertion; rerun
  under new label after completion, not a functional failure.

Still running / next:
- NumPy1.26.4 overlay model run `ci8-mtp-numpy126-r0` FAILED during Ray startup:
  the inherited PyArrow wheel explicitly requiresNumPy>=2.0. This is a dependency
  compatibility issue, not a kernel failure. Arrow22 conflicted with datasets5.1
  (requiresArrow>=24), so final constraints are pyarrow25.0.0, pandas2.2.3,
  datasets5.1.0, python-dateutil2.9.0.post0. uv r5 resolution PASSES. Arrow25 source
  retains NumPy1 support; Arrow26 lib.pyx explicitly rejectsNumPy<2.
  A no-deps overlay then missed pandas2.2's pytz dependency (diagnostic failure).
  A complete NumPy/Pandas/Arrow overlay with dependencies is now installing to
  artifacts/ci8-pyprofile (ci8-pyprofile-install-r0). run_ci8_profile.py prepends
  this to the same immutable runtime. Recheck imports then rerun MTP. Base unchanged.
- pip dry-run session15727 from04:09UTC uses envs/ci8-resolve and three indexes;
  dependency resolution progressed to downloads (slow ~330KB/s) but report not
  yet complete. logs/ci8-dependency-resolve-r0.log; report target same-label JSON.
- uv0.13.8 installed in tools/ci8-python-tools; two early attempts before install
  finished failed127 and are logged. uv pip compile requirements_ascend.txt and ascend_kernels.txt
  with constraints / official indexes, unsafe-best-match (pip-like across the
  three trusted indexes), writes a dependency solution without downloading all
  large wheels. r3 old constraints resolved but selected incompatibleArrow26;
  r4 Arrow22 was unsatisfiable; r5 Arrow25/pandas2.2/datasets5.1 PASSES and records
  the complete solution. Original pipr0 eventually completed, report exists.
- GitHub connector raw check-run status was stale. gh api with Cache-Control:
  no-cache showed unit_test COMPLETED failure; fresh log saved as
  logs/ci8-unit-github-r1.log. Use fresh gh reads where connector freshness lags.
- Final precommitr3 passes; CPU tests207passed/8skipped. Final wheel/sdist rebuilt
  into artifacts/ci8-dist-r1, source/archives inventory ci8-final-source-r0.json.
  PyArrow25/Pandas2.2 complete overlay imports and Sage MTP now PASS21/21 token/
  score bits: results/ci8-profile-model-comparison-r0.json. inventory/ci8-tested-tree-r0.json
  proves its tested working-tree diff SHA256 EXACTLY equals git diff85..111.
- Summary saved results/ci8-summary-r0.json. Old unit85 actually finished with
  3failed6424passed400skipped, same three known introduced failures, freshCLI logr1.
- New CI111 started: lint job133098612154 run43806572256; Ascend Docker
  job133098612684 run43806572399; unit job133098612569 run43806572235;
  ETEjob133098672058 run43806572228. Initial status JSONL in inventory/ci8-new-head-checks-r0.jsonl.
  Prefer fresh gh api with Cache-Control:no-cache because connector reads lagged.
- PR body DRAFT prepared artifacts/ci8-pr-body-r0.md (not yet applied). Need
  publish report/prompt first, then update PR body; fix stale external-kernel/
  CANN8.5 claims and explicitly correct3 unit-failure attribution. Continue newCI
  until relevant failures are handled; do not treat running jobs as passed.
- Publish CI/Docker report + evidence in fork docs branch, update indexes and
  PR linked report. Avoid blindly printing GitHub PR responses (nested diff huge).

GPU prompt ready at artifacts/ci8-gpu-validation-prompt.txt, needs publishing/
final link. It freezes PR/base, isolates CUDA dependencies, covers six requested
models and fixed-history precision/performance, and optional pureRay x86 control.
Final answer must include prompt or accessible copy-ready link, currentPR/CI
status, and Ray next steps (privileged host callback capture; not more identical
container replays or ARM speculation).

Use GitHub connector for metadata/PR-body update (last write succeeded). Public
repo search unsupported -> used gh api. Git push fallback that works:
GIT_CONFIG_GLOBAL=/dev/null git -c credential.helper='!gh auth git-credential' push
https://github.com/silencelamb/lmdeploy.git <branch>. Never change globalgit config.
No subagents authorized. Keep user updates within60s. New labels only.
