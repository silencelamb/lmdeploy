# PR #5134：H200 CUDA 迁移回归验证

> 执行中。以下主矩阵优先于后面的历史阶段记录；性能尚未完成，不是整项通过声明。

## 当前主矩阵（21:55）

固定共同基线 `442097e83799c2e85df9d6bbce35c7c79ea05dfe`，当前PR `f64ab5061153b5c5b63721cf2d8d6f1b41b95104`。所有数值通过均限定于保存的相同CUDA依赖、GPU与Triton调优缓存。

| 完整模型 | 主验收硬件 | 非MTP | MTP3 | 当前限制 |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash | 单机8×H200，TP8 | 固定70000 cache blocks独占重跑：4配置、双侧2次全新进程通过 | 4配置、双侧2次全新进程通过；实际cache容量一致 | 初轮TP8非MTP资源重叠记录已作废，不计主通过 |
| Intern-Decision-4B | 单H200，TP1 | 用户路径、固定16000 blocks独占重跑：4配置每轮46响应/57960候选分数全部逐bit一致 | 任务不支持 | 评分不生成token；仍需完成真实prefill graph重放审计 |
| Sage-A1.5 | 单H200，TP1 | 4配置、双侧2次新进程通过 | 同左 | 非MTP pipeline/HTTP及MTP3 pipeline各5对完成，未触发持续回退判据 |
| Agents-A1 | 单H200，TP1 | 4配置、双侧2次新进程通过 | 同左 | 非MTP/MTP3 pipeline未触发回退；HTTP C4 TTFT触发，定位中 |
| Intern-S2-Preview-397B-FP8（用户确认版本） | 单机8×H200，TP8 | 4配置、双侧2次新进程通过 | 同左 | 实际cache容量一致；非MTP pipeline五对完成；HTTP C4基线自身token不重复 |
| GLM-5.3 | 单机8×H200，TP8，block64 | 21个短/批/混批响应一致；32K长上下文基线/PR均不自重复 | 同类长上下文自重复失败 | 基线默认sparse top-k无序输出问题；不计全模型零容差通过，不做性能通过结论 |

生成模型每种配置每轮23响应、1408 token、7040完整top-5数值；四配置为eager/graph×prefix off/on。新主矩阵机器可读结果保存在本机 `results/primary-matrix.json`，最终归档会更新。原TP4 Flash/S2仅作为补充，不能替代用户要求的TP8。

主矩阵重新从176个完整运行的结果及命令/原始日志生成；44个配置的物理GPU、实际cache容量、输入长度和cache命中条件均核对一致。36个配置逐bit通过；GLM的8个配置保留基线和PR均不自重复的结论。Decision的graph配置分数已对照通过，但非空CUDA kernel图的真实重放仍单独审计，不能只用配置标志声称覆盖。

### 已完成性能样本（仍有补充测量和定位未结束）

每格5对AB/BA，输出256 token，graph/prefix off。可比条目要求双方输入/输出长度、生成token IDs和cache命中条件一致；不满足者不作迁移性能通过结论。表内为每侧中位数，括号为**逐对百分比差的中位数**；两者计算方法不同。TTFT/TPOT单位为ms。完整逐轮值、置信区间、采样显存和token记录见本机`results/performance-raw.json`、`performance-summary.json`，最终分卷会更新。HTTP是实际loopback请求；pipeline不含HTTP/router。

| 模型 | MTP | 范围/C | TPS base→PR（配对Δ） | TTFT base→PR（配对Δ） | TPOT base→PR（配对Δ） | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| Agents | 0 | http/C1 | 152.90→153.03 (+0.00%) | 66.88→65.76 (-1.82%) | 6.30→6.30 (+0.02%) | 可比；未触发持续回退 |
| Agents | 0 | http/C4 | 532.78→435.36 (-17.63%) | 124.81→580.86 (+324.32%) | 7.04→6.93 (-1.36%) | 触发回退判据，定位中 |
| Agents | 0 | pipeline/C1 | 153.08→153.12 (+0.03%) | 65.44→66.08 (+0.95%) | 6.30→6.30 (-0.02%) | 可比；未触发持续回退 |
| Agents | 0 | pipeline/C4 | 557.73→557.32 (-0.09%) | 69.48→70.56 (+0.99%) | 6.93→6.93 (+0.02%) | 可比；未触发持续回退 |
| Agents | 3 | pipeline/C1 | 107.40→107.19 (-0.11%) | 75.20→75.62 (+0.30%) | 9.05→9.07 (+0.11%) | 可比；未触发持续回退 |
| Agents | 3 | pipeline/C4 | 358.66→357.87 (-0.24%) | 78.56→80.13 (+2.04%) | 10.89→10.90 (+0.13%) | 可比；未触发持续回退 |
| GLM-5.3-Flash TP8 | 0 | http/C1 | 81.80→82.28 (+0.12%) | 181.08→181.40 (-0.55%) | 11.55→11.51 (+0.18%) | 可比；未触发持续回退 |
| GLM-5.3-Flash TP8 | 0 | http/C4 | 282.95→300.32 (+1.64%) | 356.18→243.27 (-19.16%) | 12.91→12.45 (-1.87%) | 生成token不一致，性能不计通过 |
| GLM-5.3-Flash TP8 | 0 | pipeline/C1 | 77.63→78.43 (+1.12%) | 187.10→189.85 (-3.35%) | 12.14→12.05 (-0.98%) | 可比；未触发持续回退 |
| GLM-5.3-Flash TP8 | 0 | pipeline/C4 | 307.48→306.47 (+2.41%) | 201.35→193.34 (-3.23%) | 12.21→12.34 (-1.70%) | 可比；未触发持续回退 |
| GLM-5.3-Flash TP8 | 3 | pipeline/C1 | 162.14→157.25 (+0.66%) | 198.10→191.58 (-4.12%) | 5.43→5.62 (-0.17%) | 可比；未触发持续回退 |
| GLM-5.3-Flash TP8 | 3 | pipeline/C4 | 605.47→568.75 (-3.11%) | 204.91→216.28 (+2.12%) | 5.32→5.48 (+2.76%) | 可比；未触发持续回退 |
| Intern-S2 Preview397B TP8 | 0 | http/C1 | 70.80→70.52 (-0.34%) | 186.77→200.71 (+6.33%) | 13.44→13.44 (+0.01%) | 可比；未触发持续回退 |
| Intern-S2 Preview397B TP8 | 0 | http/C4 | 253.37→254.00 (+0.13%) | 369.08→357.58 (-0.60%) | 14.37→14.39 (+0.13%) | 生成token不一致，性能不计通过 |
| Intern-S2 Preview397B TP8 | 0 | pipeline/C1 | 70.51→70.59 (+0.17%) | 194.10→192.02 (-3.75%) | 13.47→13.47 (+0.04%) | 可比；未触发持续回退 |
| Intern-S2 Preview397B TP8 | 0 | pipeline/C4 | 264.94→265.30 (+0.26%) | 203.75→198.00 (-4.46%) | 14.35→14.35 (-0.01%) | 可比；未触发持续回退 |
| Sage | 0 | http/C1 | 153.10→152.93 (+0.01%) | 66.89→65.91 (-1.33%) | 6.29→6.30 (+0.02%) | 可比；未触发持续回退 |
| Sage | 0 | http/C4 | 541.19→546.75 (+1.03%) | 129.10→121.53 (-6.27%) | 6.98→6.85 (-1.73%) | 可比；未触发持续回退 |
| Sage | 0 | pipeline/C1 | 153.00→153.07 (+0.05%) | 66.00→65.64 (-0.49%) | 6.30→6.30 (-0.06%) | 可比；未触发持续回退 |
| Sage | 0 | pipeline/C4 | 575.16→575.19 (+0.00%) | 68.82→69.89 (+0.90%) | 6.71→6.71 (-0.02%) | 可比；未触发持续回退 |
| Sage | 3 | pipeline/C1 | 280.53→280.49 (-0.08%) | 74.54→75.24 (+0.53%) | 3.28→3.29 (-0.02%) | 可比；未触发持续回退 |
| Sage | 3 | pipeline/C4 | 964.66→966.67 (+0.13%) | 80.48→78.75 (-2.20%) | 3.71→3.71 (+0.02%) | 可比；未触发持续回退 |

Agents HTTP C4原始五轮TTFT均值：base `[0.1248,0.1092,0.1089,0.1378,0.1373]` 秒，PR `[0.5856,0.1335,0.5809,0.1427,0.5828]` 秒。随后独立诊断在**基线**复现约0.581秒，PR预热也出现约0.593秒；第一轮诊断记录到的Triton模块加载只占4–8ms，**不能据此把半秒延迟归因于加载**。进一步记录Autotuner.run/benchmark等阶段，原始五对样本及触发结论保留。

Flash/S2 HTTP C4均有基线自身的新进程token差异，首个分歧和此前一致的历史见`results/http-first-divergences.json`；没有沿不同历史比较后续分数。已安排独立记录实际C4首个prefill的全部TP rank位置、KV/GDN/卷积状态、逐层hidden、target logits与HTTP序列化前的完整top-k ID/score bits。其时间不计入性能样本。固定批次pipeline结论独立保留。

Decision图节点审计首轮遇到辅助代码的构造参数问题：仅修改Python `__new__`未将`keep_graph=True`传入底层`__init__`，因此诊断在捕获结束时中止。已保存失败并修正辅助代码，待实际GPU补跑；这不是未修改模型路径的失败。Decision评分性能随后继续。

## 历史阶段与原始证据（保留失败和更正）

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
| Intern-S2 | 用户随后授权的 Intern-S2-Preview-397B-FP8 `b9b98067…`，96分片，TP4 | 完整文件已验证，整模型对照执行中；版本明确绑定 Preview |
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

已发布阶段证据：[文件/SHA256清单](evidence/cuda-h200-20261116/manifest.json)，
[分卷1](evidence/cuda-h200-20261116/evidence.tar.gz.part001)、
[分卷2](evidence/cuda-h200-20261116/evidence.tar.gz.part002)、
[分卷3](evidence/cuda-h200-20261116/evidence.tar.gz.part003)。下载后按编号拼接再解压。
归档内每个文件均按归档实际字节重新核对 SHA256；超过阈值的完整逐层 tensor 留在本机，清单明确标为 `published=false` 并给出本机路径与哈希，未声称它们已上传。

### 14:20 阶段更新

- Sage-A1.5 / Agents-A1：`matched-f64` 的非 MTP 与 MTP3，均完成 eager/graph × prefix off/on 四配置；每配置各两次全新进程，双侧自重复和两组跨源码对照全部逐 bit 一致。每模型共 8 配置，每配置每轮 1408 token / 7040 分数。prefix 长请求重放实测命中 32512 个 token。
- 所有已完成配对实际分配的 KV block 数（不仅配置的 fraction）亦一致，见实际 cache 审计。
- 未修改的基线整模型使用原 PR 的缓存，可复现原 PR 全部 1408 token / 7040 分数；原始跨缓存差异的源码无关归因已有完整模型交叉控制，不仅是小算子推断。
- 用户补充授权用 **Intern-S2-Preview-397B-FP8** 作为 Intern-S2 覆盖。固定 `b9b98067e5a0cea541b9e4a2739c0aa7a5ddb667`，96 分片约408 GB，完整文件 SHA256 已保存；TP4 使用 GPU0–3，正在执行完整模型矩阵。该结果绑定 Preview 版本。
- Decision CPU/协议合同另在共同基线与当前 HEAD 各执行：281 passed、9 skipped；不作为完整权重验证。
- 算子性能：SiLU/sigmoid × rows1/128/864/4096 × eager/graph，每格5对AB/BA，共160原始样本。各格中位配对差值约−1.46%至+0.22%，无持续>5%；eager包含Python/launch，graph为100次调用图的重放平均。这不替代整模型性能。

## 用户补充的主验收拓扑

用户要求几百B模型统一单机8卡。本轮后续主验收使用 GLM-5.3-Flash / Intern-S2-Preview-397B-FP8 / GLM-5.3 的 **TP8，GPU0–7**。前述Flash/S2 TP4结果及HTTP试运行只保留为补充证据，不能代替TP8主矩阵或正式性能。TP8重跑使用独立结果名 `matched-f64-tp8`，不覆盖TP4数据。Sage/Agents及Decision的TP1保持不变。

GLM-5.3首次通用block128配置被两版本同一FlashMLA校验拒绝，未进入推理；已保存到`failures/glm-block128`并统一按要求设为64。有效TP8配置的短/批/混批21响应全部一致，但32K长上下文首token分数在**基线与PR各自两次新进程**中也不重复。该项目前是基线重复性问题，不能计通过。已启动所有TP rank的固定chunk KV/位置/indexer/逐层hidden/target-logits采集，保持默认算法，不以确定性模式覆盖原始失败。

## 19:15 资格审计与更正

调度器重启后有一个未随父控制器退出的子队列继续运行。后续审计发现，**初轮Flash TP8非MTP及初轮Decision的实际KV容量不一致**。其数值相等结果保留，但撤销正式合格对照结论；不能用它们满足用户的相同cache条件。原始文件与精确受影响清单保留在`resource-invalidations.json`，没有覆盖或删除。Sage/Agents TP1、此前TP4补充、GLM TP8基线问题诊断、Flash TP8 MTP3、S2 TP8矩阵未受该重叠影响；后两者的每组实际cache block数已核对一致。

启动器现对每个物理GPU使用互斥锁（由子进程继承），同名运行禁止覆盖，并在启动前核对显存空闲。正在独占重跑：Decision单卡TP1、固定16000个block、用户指定目录；Flash非MTP单机TP8、固定70000个block。新结果分别使用`matched-f64-decision-isolated`和`matched-f64-tp8-isolated`标签，并记录实际生效配置。此修正仅控制实验资源与cache容量，未改变模型算法、同步次序或零容差判据。

### GLM长上下文已获得冻结输入的因果证据

所有8个rank的第二个长prefill chunk均已保存。两边input IDs、position IDs、layer0 Q/K/V、有效index score、选中索引集合一致，但索引顺序不同；layer0 attention输出开始不同，后续逐层hidden及target logits不同。

用真实冻结scores、KV和query重放未修改的默认算子：32次selector产生32种索引顺序，集合均相同。分别固定两次原始运行实际产生的顺序，attention各重复3次均逐bit稳定，并分别精确复现原始attention输出。基线和PR源码均复现同一规律，**没有开启deterministic模式、排序后再送入attention、修改kernel或放宽阈值**。原基线的`sparse_index_topk.py`使用原子输出偏移，源码已注明顺序会影响BF16 sparse attention的softmax分块。该问题归属于基线默认算法的长上下文重复性，未混入迁移PR修复，GLM长上下文不计通过。

### Decision固定本机权重

用户指定路径为`/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--internlm--Intern-Decision-4B`。21个文件已实测SHA256；模型运行所需的19个文件与已下载官方`38c9dafb…`内容一致，仅README和.gitattributes不同。重跑和后续性能直接使用用户目录。先前首轮数值一致但资源控制不合格的记录仅作历史证据；图配置出现空图警告，已安排独立的真实请求图重放/DOT核查，不仅凭配置名称声称graph覆盖。
