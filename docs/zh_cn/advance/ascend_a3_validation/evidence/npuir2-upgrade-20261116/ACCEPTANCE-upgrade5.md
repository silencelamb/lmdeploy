# AscendNPU-IR 2.0 upgrade validation, 2026-11-16 (Asia/Shanghai)

Keep the fixed native environment and migration PR untouched. Create a new local
environment with independent Triton/compiler files; reuse unchanged dependencies
read-only. Do not install or import dlinfer. Use the existing startup import guard,
selected LMDeploy checkout as cwd, and separate compiler caches for new labels.

Baseline: native LMDeploy 5f1b1b3c12b1c198813f65275f778c7ad292fb02,
research workaround f9ab50d877c384bfb04ae43afa6af0c388fc451b,
Python 3.12.13, Torch 2.10.0+cpu, torch_npu 2.10.0.post4,
Triton-Ascend distribution 3.2.2, bundled AscendNPU IR 1.2.0
042c923d02a22e5845cf10311cf2d192b17a5d77, CANN runtime 9.1.0.

1. Verify ancestry: v2.0.0 contains 05f2adcf; record actual binary versions,
   hashes, downloaded package URLs, checksums, and compiler executable selected
   by Triton. A version string or a pip metadata upgrade alone is insufficient.
2. Prefer an isolated compiler-component update, keeping CANN runtime and
   Torch/Triton frontend fixed. Official recommended pairing is NPUIR 2.0.0 with
   CANN 9.2.0; compatibility with this machine's CANN 9.1.0 is experimental.
   If needed use the matching extracted BiSheng backend or a new official
   Triton-Ascend wheel, in independent directories. Record each failed pairing.
3. Correctness gate: the standalone CPU-oracle recurrence reproducer must have
   zero corrupt state iterations with num_stages=3, including restored inputs.
   Run num_stages=1 control, real saved GDN fixture, and the six strengthened
   Q=1/2/4 eager/graph hardware cases. Force stages=3 explicitly where necessary;
   the LMDeploy single-buffer workaround must not hide a broken compiler.
4. Model gate after kernel correctness: two fresh seeded Sage-A1.5 TP2/MTP3,
   graph/prefix/BF16 runs, trace21 scenario, 64 generated tokens, compare all
   21 token sequences and top scores. Do not compare logits after histories
   have diverged. Preserve individual run/source/environment records.
5. Performance after correctness: repeat the GDN paired AB/BA matrix with
   correct stages=1 versus correct stages=3. If full-model validation succeeds,
   measure the frozen-input full target backbone using the established v2
   restoration harness. Separate operator/backbone numbers from HTTP throughput.
   Exclude CPU18 from benchmark process affinity to reduce already-diagnosed
   host interference; this is not a permanent Ray/host fix.
6. Only claim the configurations and shapes actually tested. Compiler build,
   runtime/driver, API incompatibility, model launch failure, and any unresolved
   state error are separate failures, never passes or unexplained BF16 noise.

Save commands, logs, package manifests, raw results, failed attempts and this
plan; publish a new evidence section on the fork documentation branch. No
upstream bug report or PR is needed for an already merged identical fix.
