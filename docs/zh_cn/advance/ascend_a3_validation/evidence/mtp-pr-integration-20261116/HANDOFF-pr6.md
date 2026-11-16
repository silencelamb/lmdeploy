# MTP fix integrated into native migration PR

2026-11-16 Asia/Shanghai; raw logs UTC. The user explicitly requested that the
migration PR include these fixes and describe the symptom/commits/compiler
upgrade/results/report link. This supersedes the earlier instruction to keep
the GDN fix out of that PR.

- PR: https://github.com/InternLM/lmdeploy/pull/5134
- New implementation commit: 85a0581228ad33a695430a1f43a47358a8258131.
- Integration worktree: `lmdeploy-pr6`, branch
  `research/ascend-mtp-pr-integration-20261116`, based on original PR5f1b1b3c.
- Formal PR branch: `pr/ascend-a3-native`. Its previous stage3-only HEAD5f1b1b3c
  is historical now; do not assume the PR checkout still represents that control.
- Runtime change: `num_stages = 1 if cache_seqlens_rb is not None else 3`.
  The circular path uses a conservative default on old and new compilers;
  other recurrent paths keep3. No version/shape autotuning was introduced.
- Ported research dfa3049f + d2d933da state regression. Comment attributes the
  specific NPUIR042c923d error and upstream05f2adcf included in2.0.0.
- Added optional `requirements/ascend_triton_npuir2.txt` (3.2.2.post2) and
  bilingual guide. The old CANN8.5 Docker recipe remains unchanged; this optional
  component update is for an existing compatible, independently owned environment.
  No fresh dependency solve, full CANN9.2 runtime/driver, or Docker certification.

Final commit checks: applicable pre-commit hooks passed; CPU contracts32pass/
1skip; old compiler1.2.0 and new compiler2.0.0 each6/6 hardware cases passed,
with launch audit confirming circular stage1. Two fresh Sage TP2/MTP3 graph/
prefix runs per environment each matched21 responses,1344 tokens,6720 top5
score values bitwise. See results/pr6-summary-r0.json and all pr6 labels.

Old/new dependency environments were not modified and do not contain/import
dlinfer. Prior performance measurements remain attributed to their original
commits/profiles; this integration did not rerun performance. Single buffering
won4/6 new-compiler microbenchmark shapes, but is not universally faster.

PR body snapshot before update: inventory/pr6-pr-before-r0.json.
Updated body: artifacts/pr6-pr-body.md. All external updates must be verified;
final code/docs/PR-body hashes and remote status are in
inventory/pr6-completion-r0.json after publication finishes.

CPU18 exact tasklet identity/permanent host fix remains pending privileged
host capture. No CPU affinity workaround or Ray await/order changes are shipped
in the PR. Detailed evidence stays on the fork documentation branch, not in the
migration implementation. Research host tools remain in the earlier branch.
