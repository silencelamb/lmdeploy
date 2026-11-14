# ACCEPTANCE-EXTENSION.txt

原始验证文件快照；源码 SHA256：`a3ade9c116418e3bd81f5d062bf627248c19e0346297de3fb531245eb6583f0b`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```text
Added 2026-11-12 before Agents-A1/Mooncake measurements, per explicit user request.

Keep the original pinned comparisons and environments intact.
Agents-A1 uses /afs-weights/Agents-A1 and its A3 deployment document: TP2 BF16,
session262144, block128, batch16, prefix state budget16/alignment128/last-only,
MTP OFF (the provided Agents-A1 deployment does not enable it).
Run paired eager traces, graph/prefix with reference answers and fixed-history
scores, near-limit long retrieval, and warmed HTTP C1/C4/C16 performance.
All original token, score and >5% repeatable performance gates still apply.

Mooncake uses the user-requested 0.3.13.post1 NPU distribution, not the image's
0.3.11.post1 or a CUDA wheel. Record wheel SHA256 and actual transport libraries.
Follow the GLM deployment's embedded MooncakeStoreConnector/kv_both design,
adapt transport/config to actual Ascend capabilities; do not use the remote
master address found in /afs-weights/mooncake.json. Use owned local masters.
Validate actual NPU-buffer save/load, data equality, external lookup/hit/load
evidence and model result parity, not only installation or process startup.
Separate local prefix hits from Store hits using an engine with no local cached
prefix while producer/Store segments remain alive. Record cold/warm latency.
Compare Sage-A1.5 (MTP3) and Agents-A1 (MTP off) before/after migration, with
identical model/config per pair. Test cross-request state isolation for hybrid
GDN state plus attention KV. Keep failed loads and fallback/miss visible.
If original main already rejects Ascend Store configuration, record it as a
baseline limitation. Any common platform-enabling adapter must be isolated on
separate branches and applied equally; never relabel that as untouched-main
success. No algorithm or model optimization is authorized by this extension.
Report multi-node/remote RDMA and sustained production capacity as unverified
unless actually exercised. The document's GLM/H200 benchmark is not an A3 result.

```
