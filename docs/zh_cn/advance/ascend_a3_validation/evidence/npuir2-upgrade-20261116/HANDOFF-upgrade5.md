# AscendNPU-IR v2.0.0 isolated upgrade (validation complete)

2026-11-16 Asia/Shanghai; raw logs UTC. This extends HANDOFF-ir2.md and the
upstream4 source audit. Do not replace the original native environment or PR.

## Fixed source and environments

- Independent branch `research/ascend-npuir2-upgrade-20261116`, checkout
  `lmdeploy-upgrade5`, based on original native `5f1b1b3c12b1c198813f65275f778c7ad292fb02`.
  It has no circular-state single-buffer workaround, so GDN uses stages=3.
- `envs/upgrade5-ir2`: original Triton frontend, only its private compiler bundle
  replaced with the official 2.0.0 bundle. Existing dependencies reused by
  symlink; **never pip install/uninstall through these experimental environments**.
- `envs/upgrade5-full322p2`: complete official Triton-Ascend 3.2.2.post2 package,
  same unchanged Torch 2.10.0+cpu / torch_npu 2.10.0.post4 / CANN runtime 9.1.0.
- Both use guard/sitecustomize.py; no dlinfer installed or imported. Invoke
  through scripts/run_upgrade5.py from selected source checkout. CPU18 excluded
  for test processes; no host configuration changed. Separate compiler caches.
- Official wheel URL and SHA256 are in `artifacts/upgrade5-downloads/*.whl.json`.
  Extracted bundle `artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir`.
  Compiler is **2.0.0 / efb7141581c0**, the exact v2.0.0 tag; hivmc is also2.0.0.

## Confirmed so far

- Original 1.2.0 compiler 042c923d comes from merging a3_fix_kda into feature/0725.
  It and v1.2.0 do not contain 05f2adcf; v2.0.0 does. Current official recommended
  pair is NPUIR2.0.0/CANN9.2.0; the tested component upgrade keeps CANN9.1.0.
- Original standalone stage3: 100/100 corrupt history. New component stage3:
  0/100, maxabs0. New component stage1: 0/100, maxabs0.
- Real fixed GDN fixture, stages3/graph, new component: 200/200 state and output
  bitwise identical to correct reference, maxabs0.
- Strengthened six Q1/2/4 eager/graph hardware tests: 6/6 on both component-only
  and complete Triton3.2.2.post2 upgrades, using unpatched stages3 kernel source.
- IR proof: `%94` state buffer and `%110` MTE3-to-Vector event both derive from
  inner-loop counter `%88`, store reads `%94`, loop yields `%94`.
  See inventory/upgrade5-ir-verification-r0.json and saved final HIVM IR.
- Both component-only and full-package MTP trace21 r0/r1 completed: each run's
  21/21 sequences and all top5 logprobs exactly match the earlier correct
  single-buffer result mtp-stage1-suite-r0.json (84 responses across 4 runs).
- New full-package GDN AB/BA matrix: both stages3 and stages1 correct at every
  B1/4/16,Q2/4 shape. Stages3 versus stages1 median time changes are -0.28%,
  +1.51%, +2.20%, -2.51%, +1.44%, +4.83%; multibuffering is not always faster.

## Final performance and remaining scope

- Sequential frozen target v2 harness completed, old correct single-buffer
  versus full-package new stages3, order old/new/new/old on devices0,1.
  Old/new median ms by pair/rank: 12.939/13.273, 13.136/13.095,
  13.357/13.065, 13.364/13.214. Changes +2.58%, -0.31%, -2.18%, -1.12%.
  All inputs/positions/history/initial cache hashes match; all 48 replay output
  hashes match within/across corresponding runs. No consistent slowdown, but
  these are correlated TP-rank views and do not establish zero regression or
  HTTP throughput. See results/upgrade5-summary-r0.json.
- CANN9.2.0 toolkit downloaded solely as a component fallback; its MD5 matches
  the official OBS ETag and SHA256 is saved. It has not been installed or used.
  The initial slow serial CANN download was cancelled and its partial file
  retained; completed ranged r1 download is the usable package. Do not install
  global CANN/driver.
- Preserved every label, command/environment JSON, full stdout, package hashes,
  and new source audit. Final report/evidence are on fork docs branch; final
  commit and remote verification are recorded in inventory/upgrade5-completion-r0.json.

Read ACCEPTANCE-upgrade5.md before further comparisons. Full CANN9.2 runtime,
driver upgrade, HTTP throughput, other models/shapes are not validated here.

No source-level compiler rebuild or isolated cherry-pick was tested: the official
v2.0.0 binary includes many changes beyond 05f2adcf. Its actual final IR and state
behavior are correct for these checks. New compiler correctness does not imply
multibuffering is universally faster. Native migration PR remains untouched.
