# CANN9.1 Docker/dependency alignment and PR5134 CI repair

User requests: keep CANN9.1.0; align Dockerfile/dependencies, audit PR5134 CI,
fix regressions introduced by this PR, separately record inherited problems,
explain remaining Ray work, and provide a self-contained GPU validation prompt.
Do not resume the cancelled CANN9.2 upgrade or modify the active runtime.

Start from PR85a05812 in independent worktree lmdeploy-ci8. Pin upstream
comparison base442097e8. Save current failed CI logs and exact job/head IDs.
Before changing a failure, compare its code path and reproduce on base/PR with
the same available environment. Missing hardware is not a passing control.

Known work at start:
- Two dead links introduced in the Ascend kernel README: replace with verified
  immutable source/provenance links; do not weaken the link checker.
- Three model-agent sampling tests: base3pass/PR3fail reproduced on CPU, caused
  by new cache_config access. Preserve CPU/CUDA semantics and real Ascend FP32
  conversion; rerun the affected test file and native model traces.
- Ascend Docker GitCode clone returns418. External kernels metadata also pins
  old Torch/Triton. Import only the already-used, already-validated MIT GDN
  prefill/L2Norm slice (11files,1670lines) with provenance and existing snapshot
  compatibility edits, then remove that floating Git dependency. Validate
  executable AST/provenance and real outputs against the prior external slice.
- Docker target: official CANN9.1.0 A3/Python3.12 image, Torch2.10.0+cpu,
  TorchNPU2.10.0.post4, torchvision0.25.0+cpu, Triton-Ascend3.2.2.post2. Resolve
  compatible remaining dependencies without globally replacing CUDA requirements.
- Jetson libcudss.so.0 import failure: audit unchanged Docker/requirements and
  CI evidence; keep inherited/externally caused issues separate from native fixes.

Run applicable lint/link/package/dependency checks; CPU base/PR/fixed controls;
GDN prefill equivalence, six recurrent state tests, and representative seeded
model traces on CANN9.1 without dlinfer or external-kernel imports. Use fresh
labels and isolated dependency environments. Recheck new-head CI after pushing
validated changes; report running/unverified jobs explicitly.

GPU prompt must pin both PR head and common upstream base, keep CUDA stacks and
checkpoints identical, forbid Ascend dependencies, cover the requested six-model
matrix and shared RMSNormGated paths, compare fixed histories for precision,
separate baseline nondeterminism/failures and measure matched performance.

Update the existing PR and fork docs branch. No host IRQ/affinity workaround,
Ray synchronization reorder, CANN9.2 update, or unrelated algorithm optimization.
