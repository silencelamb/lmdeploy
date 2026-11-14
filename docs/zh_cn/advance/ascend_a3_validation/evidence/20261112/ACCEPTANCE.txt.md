# ACCEPTANCE.txt

原始验证文件快照；源码 SHA256：`4ff6b37cd5a2c55a25be9e82542c5297d627ec3663b593649e880d96349eb5c4`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```text
Predeclared acceptance, 2026-11-12 UTC (before running baseline or native inference)

Pinned references:
LMDeploy baseline fb5de87a45df9999a19a569ed9441df8edfa6975
dlinfer baseline 3b9096341a41e9d067104a39d914e47ca2d7ec50
LMDeploy migration 39dbaa841080d9b9b6b0326197b9de7fe9495a1b
dlinfer cleanup 05ede38ae879347be33c8f6ebb136d84f3f4ee88
Both migration commits have the corresponding baseline as their direct parent;
the fork main refs equal those parents at initial inspection.
Any repair is made only on codex/ascend-a3-validation-20261112 worktrees.

Environment:
Use the user-confirmed deployment container and freeze its actual dependencies.
Three isolated venvs: old LMDeploy+dlinfer; native LMDeploy with no dlinfer;
standalone cleaned dlinfer with no LMDeploy. Shared dependencies are snapshotted,
not upgraded. Native imports are guarded, including spawned worker processes.
Python 3.12 and torch 2.10 fall outside some package metadata constraints;
installation overrides and pip-check conflicts must be reported explicitly.
External triton-ascend-kernels is 8e70a44afa537fcbcdc6d59e6d2196a38f1e7b6f
plus existing colleague changes; preserve a separate snapshot, patch and hashes.

Primary scenarios:
Sage-A1.5 BF16 TP2, same two devices, block size 128, batch limit 16, MTP 3,
prefix caching/aligned state saving as deployment document. Compare eager,
graph, prefix reuse, MTP off/1/3, mixed prompt lengths, repeated requests,
multilingual/math/code fixed prompts, long-context retrieval, concurrent 1/4/16.
Exercise up to 262144-token session capacity if the baseline can run it.
Supplement with Qwen3-8B dense GQA TP1 and NPU reference operator tests.
Standalone dlinfer: import isolation, attention/RoPE/sparse kernels and MoE
eager/real graph capture. Additional model families are reported only if run.

Functional gate:
No new exception, hang, invalid/empty response, nonfinite score, lost request,
cache/state cross-request contamination, or silently skipped required check.
Record baseline defects separately; identical baseline failure is not a pass.

Numerical gate:
Identical configuration, prompts, tokenization, seed and request schedule:
100% greedy generated token agreement. Save token IDs and per-token logprobs.
Same-history scores: max absolute logprob difference <=0.02 and RMS <=0.005;
require same argmax, investigate first mismatch instead of relaxing thresholds.
Operator tests retain their pre-existing reference tolerances and record skips.
Cross-mode eager/graph/MTP differences are also checked on each branch; a
baseline cross-mode defect must not be misattributed to migration.
Task quality: deterministic exact-answer checks and long-context retrieval;
do not claim GPQA 86.9 or full benchmark accuracy without that dataset/run.
Random sampling completion alone does not certify distribution equivalence.

Performance gate:
Use identical model/config/devices and fixed input/output token counts. Warm up
each mode; take >=3 measured repetitions. Record wall throughput, TTFT, output
tokens/s, inter-token latency (with MTP burst caveat), p50/p95 and peak NPU memory
when observable. A repeatable >5% throughput loss or latency increase is a
regression; noisy/borderline differences require paired reruns, not a pass.
Separately report deployment SLO TTFT<=3s and output speed>=40 tokens/s for
the measured short-prompt workload; never extend it to all context lengths.
Compilation/loading are excluded from steady-state timing and saved separately.

Limits:
No algorithm optimizations, no changes to colleague directories or main trees.
No CUDA/other accelerator pass without hardware. No claim for unrun models,
quantization modes, full 4-instance/router deployment, disaggregation or GPQA.
All actual coverage, failures, skips and evidence paths go in the final report.

```
