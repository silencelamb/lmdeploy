# ACCEPTANCE.txt

原始验证文件快照；源码 SHA256：`43b970711ac49c5553c66cdc5e1cbe14525cc669a7cec4386db94cd61f3ecd5b`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```text
2026-11-13 follow-up: PR #5030 + user-provided A3 scripts

Keep the previous evidence archive and the original model comparison unchanged.
Pin PR head 1a09270b0835bb871be4eb4df70f9d04c8ccaec2 and apply the same three-file
patch to baseline fb5de87a and repaired native 6421cefe in new local branches.
Use isolated environments with mooncake-transfer-engine-npu==0.3.13.post1.
The native environment must neither install nor import dlinfer.

Correct the preceding probe to follow this PR: protocol=ascend, explicit device
backend=ascend, ASCEND_ENABLE_USE_FABRIC_MEM=1, global segments integer multiples
of 1 GiB, local_buffer_size=0 at Store.setup, no external engine injection and
no client-side buffer registration in fabric mode. Use the supplied CANN env
exports and ACL_OP_INIT_MODE=1. Start only a locally owned master on free ports.

SDK acceptance: two real NPU processes, independent source/destination buffers,
128 KiB/2 MiB/8 MiB data, cross-process Store write/read and byte-exact SHA256
comparison. All intended transfers must complete; initialization alone is not a
pass. Correct failed runs remain recorded, including actual arguments/errors.

Model acceptance: Sage-A1.5 TP2/MTP3 and Agents-A1 TP2/MTP off, same weights and
comparable configuration in old/new environments. Prove external Store puts,
gets and hits with fresh local KV state, and validate GDN state restoration and
output correctness; in-process prefix hits alone do not count. Same-history
top-k logprob max<=0.02/RMS<=0.005 and exact greedy tokens retain the prior gate.
Baseline MTP repeatability failures remain a separate limitation.

If actual Store transfer succeeds, measure matched cold/warm latency and cache
hit behavior after warmup, at least three runs; repeated >5% performance loss
requires investigation. If initialization or transfer is blocked, external
cache precision/performance remain unverified, not passed.

```
