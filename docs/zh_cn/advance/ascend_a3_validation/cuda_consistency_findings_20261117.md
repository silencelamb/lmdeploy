# PR #5134 GPU验证结论与三类基线一致性问题

更新日期：2026-11-17，Asia/Shanghai。

本次H200验证没有发现已证实的PR独有CUDA数值回归。176个完整运行覆盖44种配置，其中36种在固定条件下同时满足双方自重复和迁移前后逐位一致；GLM-5.3的8种配置因长上下文在基线自身也不能重复而保留失败。这里的“迁移对齐”不等于任意batch布局、独立调优缓存或全部部署配置都已通过。

这份文档集中说明三个已定位的CUDA基线问题，并追溯S2的BF16 AllReduce实现来源。完整测试与性能数据见[主回归报告](cuda_regression_20261116.md)，冻结张量、逐步舍入参考和更详细代码分析见[机制报告](cuda_determinism_followup_20261117.md)。

## 固定版本与六模型结果

- 未修改共同基线：`442097e83799c2e85df9d6bbce35c7c79ea05dfe`。
- 完整H200模型运行的PR版本：`f64ab5061153b5c5b63721cf2d8d6f1b41b95104`。
- 本次发布时经GitHub API核对的PR HEAD：`2ccf38a43f0fdc7415ce4ce9f177852bdacc9966`。完整fetch后，和f64相比只有`.github/workflows/test_docker.yml`、`docker/Dockerfile.jetson`变动；`lmdeploy`、`tests`、`requirements`、`src`的git tree对象均相同。模型数据仍标记实际测试的f64，不冒称重新测试了新HEAD，也没有更新测试基线到移动的main。[版本核对记录](evidence/cuda-pr-summary-20261117/publication-version-audit.json)。

环境为单机8×H200（每卡143771MiB）、驱动570.133.20、CUDA12.8、Python3.12.14、Torch2.10.0+cu128、Triton3.6.0、FLA0.5.2。大模型用同一组8卡TP8，单卡模型用TP1；同一checkpoint文件、BF16激活、greedy和seed42。两边独立工作树及环境共享固定依赖快照，核对实际lmdeploy导入路径，未安装/导入NPU依赖。

| 完整模型 | 实际资源 | 非MTP | MTP3 |
| --- | --- | --- | --- |
| GLM-5.3-Flash | 单机8×H200，TP8 | 固定条件逐位相等 | 固定条件逐位相等 |
| Intern-Decision-4B | 单H200，TP1；用户checkpoint | 评分任务逐位相等，真实piecewise graph确认 | 不支持该任务 |
| Sage-A1.5 | 单H200，TP1 | 固定条件逐位相等 | 固定条件逐位相等 |
| Agents-A1 | 单H200，TP1 | 固定条件逐位相等 | 固定条件逐位相等 |
| Intern-S2-Preview-397B-FP8 | 单机8×H200，TP8；用户批准的Preview版本 | 固定条件逐位相等 | 固定条件逐位相等 |
| GLM-5.3 | 单机8×H200，TP8 | 长上下文两侧均不自重复，保留为基线问题 | 同类基线问题 |

每个生成模型覆盖eager/graph × prefix off/on、单请求/批请求/长短混批、新请求重放与约32K上下文，两侧各至少两次全新进程。比较生成token ID、完整top-5集合和各项分数二进制，零容差；不是只看生成文本。Decision按真实score-only任务验证，不能用生成TPS衡量。

通过条件包括保存的匹配Triton调优缓存。最初两套缓存选到了不同FLA norm调优项；四向源码/缓存交换及整模型控制证明差异跟随缓存，原失败仍保留。这是三类运行机制以外必须写明的环境前提。

## 1. GLM-5.3：原子分配输出槽位导致索引排列不确定

调用链为GLM DSA indexer→CUDA NSA→共享TileLang `sparse_index_topk`。默认`deterministic=False`用整数原子计数器预留输出位置：计数本身精确，但线程抢到位置的次序没有固定，因此同一集合可以以不同排列输出。这里首先不是浮点atomicAdd数值累加出错，而是索引排列改变，随后attention的读取/归约分块改变。

实际冻结scores、Q、KV和位置均相同，所选集合相同；默认选择器在两源码的eager/graph各32次均产生32种索引排列和32种attention输出。启用已有`deterministic=True`作算子诊断控制后，各32次只剩1种排列和1种输出。真实长度≤2048时有直接顺序输出短路径，长prefill超过该阈值后暴露问题。

GLM-5.3当前调用没有传入稳定选项；Flash的KPool调用已经显式保序。稳定选项同时固定同分选择和逻辑索引顺序，不能用仅比较排序后的集合代替验证。原始32组长响应自重复对照均存在首位置分数差异，其中28组随后生成token也分叉。

**归类：基线问题，PR两边都有。** 稳定选项的算子级控制不是完整模型修复验收；没有在迁移PR里启用它或改判原失败。

## 2. Flash/S2：batch形状改变cuBLASLt GEMM选择

真实HTTP并发4并不保证服务端一次合成4：请求分别准备、入队，当前队列快照可能分为3+1等批次。对应的矩阵行数M随实际合批改变，即使共同请求的输入行及权重逐位相同，cuBLASLt仍可能选择不同GEMM分解。

| 首差算子 | 真实冻结形状 | profiler与输出证据 |
| --- | --- | --- |
| Flash第0层fg_a_proj | M=510对680，BF16权重256×4096 | 小形状使用Split-K及splitKreduce，大形状非Split-K；每个共同请求有20个元素不同，max-abs0.0078125 |
| S2第0层MoE gate | M=167对668，BF16权重512×4096 | 同样发生Split-K/非Split-K路径切换；共同请求29个原始路由分数不同，max-abs0.0625 |

局部算子重放不含TP通信，能精确重现捕获的实际输出；两源码表现相同。Split-K将K方向求和拆成部分结果再合并，不同求和组织在有限精度下不必逐位相同。关闭BF16减精度归约没有消除本例差异，FP32计算后转BF16也仍留有个别差异；没有据此声称一个开关能解决所有batch敏感性。

**归类：基线的形状敏感性，不能只归因于TP或RMSNormGated迁移。** 这项不表示已经量化业务准确率损失，也不把“误差小”作为零容差验收通过理由。

## 3. S2 out_proj：BF16 NCCL Ring归约缺少跨布局的统一rank顺序

实际模块是第0层GDN/线性注意力输出投影，不是词表lm_head。S2此处总输入8192维，TP8后每rank输入1024维，权重4096×1024；每rank生成完整4096维的一份部分和，再进行AllReduce。TP的输入通道分片，不能等同于AllReduce对投影输出缓冲区的分块。

用完整有效的8rank冻结输入构造一致行置换，在原8卡上调用未修改的DeepGEMM provider：

1. 本地量化/FP8 GEMM在行置换前后，按逻辑请求对齐逐位相同。
2. 原生BF16 AllReduce后，16384个元素中6736个不同，max-abs0.0078125。
3. 完全跳过GEMM，只置换已保存的局部部分和再归约，仍精确复现差异。
4. 固定布局重复稳定；eager和真实CUDA graph都匹配原始有效A/B输出。两份源码的96项保存tensor对照全部逐位相同。

Profiler确认实际内核为NCCL2.27.5的`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`。本例4×4096输出对应8个2048元素块，NCCL按一维连续地址寻址，并不是每行独立按列切1/8。不同块沿同一ring从不同rank起点累加，因此请求换行后，逻辑元素的浮点求和序列可以变化。

固定环拓扑不等于所有逻辑元素使用统一的rank起点；但这也不是哪个rank先到就先加的随机竞争。固定同一物理布局时，本次重放稳定。按实际循环rank顺序逐步舍入回BF16的CPU参考，对两种排列的全部16384个输出均逐位匹配实际NCCL结果。

**归类：基线已有的跨布局数值敏感性，非PR新增。** FP32归约诊断控制消除了此冻结反例，但不是已经实现了生产修复；固定rank序列与内部FP32累加也只能保证相同局部贡献的归约一致，不能消除前节已存在的GEMM输入贡献变化。

### 与其他模型和引擎的比较应限定具体路径

固定基线的Flash已对若干输出投影及最终MoE TP归约显式升FP32。所查vLLM/SGLang常见自定义AllReduce使用低精度通信缓冲、FP32寄存器累加后再写回；所查TensorRT-LLM fused AllReduce也默认FP32累加。但这些引擎另有NCCL及其他通信路径，不是所有模型/所有后端统一FP32。输入/通信dtype与内部累加dtype必须分别说明。

本轮重读的vLLM两阶段输入指针已经按绝对rank排列，轮转的是分发缓冲；此前将其也描述成owner-first累加的表述已更正。SGLang所查两阶段仍有owner-first路径。对外比较必须引用具体源码/配置，不能泛称“其他框架都保序”。固定外部版本、代码链接和限制见[机制报告](cuda_determinism_followup_20261117.md)。

## S2的BF16归约是谁写的

没有找到一个单独为Intern-S2编写BF16加法循环的提交。S2沿用Qwen3.5和共享blocked-FP8 CUDA投影：投影输出沿用激活dtype，本轮为BF16；公共代码直接把该tensor交给distributed AllReduce，最终由NVIDIA NCCL的BF16 Ring内核执行加法。Git历史可以确认以下实现来源，不能据此推断某个人故意选择了不确定算法。

| 历史环节 | 作者与提交 | 能确认的内容 |
| --- | --- | --- |
| 共享CUDA FP8投影 | Q.Yao / q yao（GitHub `grimoire`），[ac509e86，#2967](https://github.com/InternLM/lmdeploy/commit/ac509e860c1875450378657e8b26595860624b8b)，2025-01-08 | 已有`out_dtype=x.dtype`，然后直接`dist.all_reduce(out)`，没有通信前统一升FP32 |
| **本轮实际选中的DeepGEMM路径** | **AllentDan**，[e37a76dd，#3287](https://github.com/InternLM/lmdeploy/commit/e37a76dd325f8cebce96222112cebd7fd3c4c02e)，2025-03-22 | 引入DeepGemmLinearBlockedF8Impl，同样保持输入激活dtype输出并直接AllReduce |
| Qwen3.5 GDN out_proj | Q.Yao / `grimoire`，[3bf75ff9，#4351](https://github.com/InternLM/lmdeploy/commit/3bf75ff9fc745798f518e22c5e9e92a2bbf39d23)，2026-02-27 | 接入共享build_o_proj；随后[d4e83e16，#4470](https://github.com/InternLM/lmdeploy/commit/d4e83e160eee52a66e103cbe8f2e042854cfc993)为该投影传入FP8量化配置 |
| Intern-S2 Preview接入 | zxy（GitHub `CUHKSZzxy`），[df9b428a，#4575](https://github.com/InternLM/lmdeploy/commit/df9b428a7712b138b6fdeeabd28fc94ab03ab16f)，2026-05-09 | 将InternS2Preview架构映射到既有Qwen3.5 MoE实现，未另写该归约算子 |
| 当前AllReduce行的git blame | Q.Yao，[9f4f842b，#4702](https://github.com/InternLM/lmdeploy/commit/9f4f842bd00a8fbfed73204c08de81cdd07dc945)，2026-06-25 | 只是补`group=group`；不能把这一行的最后修改者当成BF16行为的最早引入者 |
| 共享provider生命周期重构 | Q.Yao，[ad242e61，#4830](https://github.com/InternLM/lmdeploy/commit/ad242e611a6e8f6191fbd2ce5efb20d1eb724896)，2026-08-12 | 合并公共forward，沿用既有输出dtype通信方式 |
| Flash的FP32 opt-in | qescccczmr，[b31a0a49，#4968](https://github.com/InternLM/lmdeploy/commit/b31a0a4985012feeb3db83f0489199bf21558fdb)，2026-10-16 | 加入可选tp_reduce_dtype并为Flash显式启用，保留其他调用者原有默认；不是把S2从FP32改回BF16 |

以上提交全部是固定共同基线442097的祖先。迁移PR没有修改`backends/cuda/blockedf8_modules.py`、`distributed.py`、`nn/linear/base.py`及S2架构映射文件。[提交来源与祖先关系记录](evidence/cuda-pr-summary-20261117/s2-ar-provenance.json)。

因此，如果问“本轮实际DeepGEMM投影后直接归约的路径是谁接入的”，答案是AllentDan；如果问更早共享FP8同dtype归约和Qwen3.5投影的来源，则是Q.Yao/grimoire；S2 Preview的模型接入作者是zxy。实际BF16逐步累加由NCCL实现，不能归为某个S2专用手写加法算子。

## 性能、剩余限制与后续边界

- 正确性合格的工作负载预热后做至少5对AB/BA交错，记录TPS、TTFT、TPOT及显存。合格对照未触发预先约定的持续>5%且超出噪声的回退判据；不等于每个指标都已证明≤5%。
- Agents原HTTP C4 TTFT回退定位到遗漏形状的MoE在线调优；完整预热后未再触发持续回退，但配对TTFT中位仍+7.76%、区间跨0，原样保留。Flash/S2 HTTP C4因token不自重复，不计性能通过；GLM长上下文失败也不进入性能通过统计。
- 两边共有的GDN单测失败、部分prefix miss→hit分数变化、仅插桩退出挂起等仍按原报告保留。未认证非Preview S2 FP8、用户DP4/EP8/router生产链路、多机、视觉GPU前向、MTP HTTP性能、独立target性能与长期高并发。
- 两次CPU-only Ray对照未复现>50ms join，只是有限x86对照，不证明ARM根因或宣布Ray已修复。
- GLM稳定索引与S2确定性归约的修复留在[未来独立分支任务Prompt](cuda_determinism_independent_task_prompt_20261117.md)，不混入Ascend迁移PR；本轮只发布验证与归因，没有实施这些基线修复。
