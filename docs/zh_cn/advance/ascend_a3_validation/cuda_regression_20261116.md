# PR #5134：H200 CUDA 迁移回归验证

执行中，2026-11-16（Asia/Shanghai）。本文件记录实际完成的证据；未完成项不计通过。

## 固定版本与环境

- 起始 PR HEAD：`111dbf6b4dc1c1aa9fb2fd0753313e0314858c96`。
- API base 与完整非 shallow 历史核实的 merge-base 均为 `442097e83799c2e85df9d6bbce35c7c79ea05dfe`。
- 期间 HEAD 更新为 `f64ab5061153b5c5b63721cf2d8d6f1b41b95104`；仅 Ascend Dockerfile/两份安装文档变化，CUDA 实现、测试、CUDA/common requirements 与起始 HEAD 完全相同。另建工作树，补跑当前 HEAD 单测及受控模型对照。
- x86_64，双路 Intel Xeon Platinum 8558，192 logical CPUs，8× NVIDIA H200 143771 MiB。驱动 570.133.20，系统 CUDA 12.8，Python 3.12.14，Torch 2.10.0+cu128，CUDA Triton 3.6.0，FLA/fla-core 0.5.2，NumPy 2.2.6。
- 保留用户原环境。将既有 CUDA 包复制到独立目录，移除复制件中的旧 editable 路由，两套 venv 使用同一依赖快照；显式工作树 PYTHONPATH、父进程/worker 导入审计。无 dlinfer/torch_npu/triton-ascend 安装或导入。最终 `uv pip check` 通过。
- 本机证据根：`/home/daixu/data/pr5134-cuda-20261116`，工作区入口：`/home/daixu/GLM5.3-flash/pr5134-cuda-20261116`。

## 已完成单测与数值定位

| 项目 | 起始共同基线 | PR 111dbf6b | 结论 |
| --- | --- | --- | --- |
| model-agent、相关模型加载/config 共同测试 | 298 passed, 1 skipped | 298 passed, 1 skipped | 通过；旧报告的 cache_config 三异常当前已修复 |
| 新增 norm/调用者/Ascend CPU 合同 | 新 API 不存在，使用旧 FLA/工厂作 oracle | 122 passed, 1 skipped, 1 deselected | norm 文件 83/83，含 56 个真实 CUDA/graph；一项会导入模拟 torch_npu 的合同按本任务禁令排除 |
| 相关 CUDA kernel 套件 | 123 passed, 24 skipped, 1 failed | 123 passed, 24 skipped, 同一失败 | 共有失败：GDN transposed state indices 的同一数值断言；不归为迁移回归 |
| 实际 94 层 gated norm scale（Flash 34、Sage 30、Agents 30） | 冻结旧工厂 oracle | 282 eager、564 graph replay 逐字节相等 | 算子级通过，不代替整模型 |

共同 CUDA 失败为 `test_fused_gated_delta_rule_transposed_state_indices[4-False-state_dtype0]`，两边同在输出索引 `(121,0,7,58)`，1048576 个元素中一个超出原断言容差，max-abs `0.00390625`。未修改断言、算法或源码。

### 完整模型首个分歧：固定输入后发现调优缓存混杂

Sage/Agents 初始四种 eager/graph × prefix off/on 配置均有两次全新进程自重复对照。
双方各自可逐 bit 自重复，但跨版本首请求第 0 个输出位置的 top-5 分数不等；原结果保留且不计通过。

对 Sage 首次实际 target forward 保存输入 IDs、位置/attention metadata、卷积/GDN bank、活动 KV block、逐层输入/输出及完整 target logits。
诊断运行复现各自未插桩的完整 64-token 序列和分数。首次 hidden 差异在第 0 层 gated norm：`864×128` BF16 的一个元素相差 `3.0517578125e-05`；该算子的输入、gate、scale 均 byte-identical。

旧/新源码分别交叉使用两套**原始** Triton 调优缓存，在同一冻结输入上四组重放，结果完全随缓存走：

| 源码 | 原基线缓存 | 原 PR 缓存 |
| --- | --- | --- |
| 未修改基线工厂 | 与原基线输出逐 bit 相等 | 与原 PR 输出逐 bit 相等 |
| PR 公共 norm | 与原基线输出逐 bit 相等 | 与原 PR 输出逐 bit 相等 |

两套缓存自动选择 `BT=16,num_warps=16` 和 `BT=16,num_warps=8`。新增单测与旧模型在不同先前 workload 下为相同缓存 key 选了不同核配置。
这不是“BF16 误差可忽略”的结案，也未证明 PR 改坏计算；它证明独立 autotuning 缓存下原目标没有达到逐 bit 一致。
已归档原缓存 timing JSON/编译产物和冻结 tensor。额外 `matched-f64` profile 使用原基线缓存的完整副本供双方读取；不手选 winner，不改 kernel、同步或验收阈值。该额外 profile 的结果将单列，不能覆盖原始差异或宣称跨独立编译环境的重复性。

## 六个完整模型覆盖（仍在执行）

| 模型 | 固定 checkpoint/资源 | 当前状态 |
| --- | --- | --- |
| GLM-5.3-Flash | 官方 FP8 `3f1971b7…`，62 分片，TP4 | 执行中；早期 Ray socket 路径过长已修正于验证启动器，原失败保留 |
| InternDecision | 用户指定 Intern-Decision-4B `38c9dafb…`，BF16 TP1 | hub 无完整本地权重；官方固定 revision 下载中。决策评分任务不支持 MTP |
| Sage-A1.5 | `ee49fa73…`，15 分片，BF16 TP1 | 初始 profile 分数不一致已定位到缓存；matched profile 执行中 |
| Agents-A1 | `604894d0…`，14 分片，BF16 TP1 | 同上 |
| Intern-S2 | 用户指定 Intern-S2-397B-FP8 `31b460c8…`，95 分片约 406 GB | **未验证**：指定 hub 中仅有 BF16/Preview FP8；不以其替代。外网下载持续断流，已询问精确本地路径 |
| GLM-5.3 | 官方 FP8 `e0b07fd2…`，141 分片，计划 TP8 | 待执行 |

已对四个现有完整 checkpoint 的全部 271 个文件实际读取并计算 SHA256，总计约 1.226 TB；读取前后 size/mtime 稳定，232 个 safetensors 的实测 SHA256 与 HF blob 名一致。

所有生成对照固定 BF16 激活、greedy/top_k=1、Python/NumPy/Torch/CUDA seed42、batch8、session65536、prefill2048、cache fraction0.5。每轮含单请求、8请求批次、长短混批、同 prompt 新请求重放、约32K实际输入长上下文及重放；每轮23响应/1408生成token/7040 top-5数值。源 dtype 保持不变，返回数值额外保存 FP32/FP64 hex；在首个历史或分数差异处停止精度比较。

## Ray 独立 CPU 对照

独立无 Torch/NPU venv：Python3.12.14 / Ray2.48.0 / NumPy2.2.6。原报告 `ray_deep_probe_v2.py` 原样运行，两次全新进程各6000次，保持 submit→sleep12ms→join→free，无 CPU18 亲和性规避。
两轮 put 的最大 join 分别45.123/40.834ms，direct分别1.444/2.078ms；无 >50ms join。保存 driver/worker maps、投递/确认/反序列化/event-loop heartbeat、Ray C++ event logs、线程 schedstat/状态/CPU 及 `/proc/stat`、softirq时间线。
这只是有限 x86/H200 主机窗口的未复现，**不能认定 ARM 根因或宣布 Ray 问题解决**。首次临时路径过长导致的启动失败单独保留；成功的 CPU 对照与 GPU 模型性能计时完全分开。

## 性能、修复与证据

判据在运行前写入 `ACCEPTANCE.md`：数值零容差通过后，至少5对预热后AB/BA，持续 >5% 且超出配对噪声需定位；区分算子/target/pipeline/HTTP。当前尚无合格完整性能结论。未向实现 PR 添加报告或调优规避，未自动合并 PR。

后续将追加最终矩阵、逐轮性能与发布证据索引。
