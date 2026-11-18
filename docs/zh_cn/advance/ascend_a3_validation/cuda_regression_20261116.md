# PR #5134：固定版本的 H200 CUDA 迁移回归验证

执行窗口：2026-11-16 至 2026-11-18，Asia/Shanghai。原数值矩阵与11月18日性能复测分别记录。结果包含**条件通过、基线问题和未验证项**。在有效、相同条件的未插桩迁移对照中，没有发现已证实的 PR 独有 CUDA 数值回归；不能将本报告读作任意缓存、动态合批或全部部署配置的逐位一致保证。

2026-11-17进一步归因见[原子索引、动态合批与TP归约补充报告](cuda_determinism_followup_20261117.md)：已通过完整8卡冻结算子重放把本次S2 out_proj差异定位到BF16 NCCL Ring归约；旧错步数据仍排除，原主矩阵不改标为无条件通过。

面向PR审阅的[集中说明](cuda_consistency_findings_20261117.md)汇总三类问题及S2实现来源。本次发布核对HEAD为2ccf38a4，与实际已测f64仅有Jetson Docker/CI差异，推理、测试及Python依赖源码树一致；未重新标记历史运行SHA。

后续独立控制已完成[GLM-5.3保序选择器整模型验证](glm53_stable_selector_fullmodel_20261117.md)：在基线与f64各自仅启用既有deterministic=True，32次新进程的8组匹配配置通过全部自重复/跨源码比较；prefix miss→hit的共有生成top-5变化仅作跨条件现象记录，不继续排查。下文未修改默认路径的原始36/44结果保留，不改标为控制分支结果。

## 固定版本与判定范围

- 共同基线：**442097e83799c2e85df9d6bbce35c7c79ea05dfe**。GitHub API base 与完整非 shallow 历史的 merge-base 均核实为此 SHA，从未用移动中的 main 替代。
- 本轮固定 PR：**f64ab5061153b5c5b63721cf2d8d6f1b41b95104**。起始观察到111dbf6b4dc1c1aa9fb2fd0753313e0314858c96；f64仅修改Ascend安装/镜像检查，运行代码、测试和CUDA依赖文件相同。另建工作树，并在f64完成主模型矩阵和关键单测。
- 后续观察到b547ca0b89444dc6cb1db33b881e1d06aea60a4b，完整抓取后记录了Jetson cuDSS和CI fail-fast两处变更；运行代码仍相同。用户随后要求固定版本，本轮保持442097/f64，不把数据改标为b547，也不再追随新HEAD。
- 两个测试工作树始终独立、detached、无实现修改。没有向实现PR提交报告、算法改动或CPU亲和性规避，没有合并PR。

所有主数值通过都限定于**相同CUDA依赖、同一checkpoint文件、同一组物理GPU，以及保存的匹配Triton持久化调优缓存**。初始独立调优缓存下的差异没有被删除；其因果对照见下文。

## 六个完整模型的主矩阵

每个生成模型的非MTP与MTP3分别运行eager/graph × prefix off/on四种配置，每配置双方各两次全新进程；检查双方自重复和两组跨源码配对。生成模型每轮23响应、1408生成token、7040完整top-5数值；Decision评分任务每轮46响应、57960候选分数。表中的“前后逐位相等”只指相同事件、相同实际缓存/合批条件的迁移对照；它不代表该模型在独立调优缓存、任意动态batch或跨cache命中状态下也确定。

| 完整checkpoint | 硬件/TP | 非MTP：固定条件迁移对照 | MTP3：固定条件迁移对照 | 不能据此前后相等宣称通过的范围 |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash，官方FP8 | 单机8×H200，TP8 | 前后逐位相等：4配置、双侧双重复 | 前后逐位相等：4配置、双侧双重复 | HTTP C4动态合批不自重复；prefix长请求miss→hit的分数变化两边都有 |
| Intern-Decision-4B，用户目录 | 单H200，TP1 | 前后逐位相等：4配置、双侧双重复；真实piecewise graph确认 | 任务不支持 | score-only，不能用生成TPS/TTFT/TPOT衡量；部分prefix长评分miss→hit分数变化两边都有 |
| Sage-A1.5，完整BF16 | 单H200，TP1 | 前后逐位相等：4配置、双侧双重复 | 前后逐位相等：4配置、双侧双重复 | prefix长请求miss→hit分数变化两边都有 |
| Agents-A1，完整BF16 | 单H200，TP1 | 前后逐位相等：4配置、双侧双重复 | 前后逐位相等：4配置、双侧双重复 | prefix长请求miss→hit分数变化两边都有；最新HTTP性能见11月18日复测 |
| Intern-S2-Preview-397B-FP8，用户明确批准的版本 | 单机8×H200，TP8 | 前后逐位相等：4配置、双侧双重复 | 前后逐位相等：4配置、双侧双重复 | HTTP C4对实际批量形状/请求排列敏感；不能替代未找到的非Preview FP8版本 |
| GLM-5.3，官方FP8 | 单机8×H200，TP8 | 基线问题：32K长上下文双方均不自重复 | 同类基线问题 | 之前21个短/批/混批响应一致；长上下文不满足零容差，不进入性能通过统计 |

主矩阵来自176个完整运行，共44种配置；36种满足全部数值比较，GLM的8种保留失败。所有44种均重新核对物理GPU、实际KV容量、输入长度及cache命中条件一致。[可读主清单](evidence/cuda-h200-20261116/primary-summary.json)对应归档中的完整primary-matrix.json。

原Flash/S2 TP4只作为历史补充。一次控制器重启留下子队列，造成初轮Flash TP8非MTP与初轮Decision的显存重叠和实际KV容量变化：这些轮次已**撤销主验收资格**，原始文件和明确排除清单resource-invalidations.json保留。随后使用继承到子进程的GPU互斥锁、启动前空闲显存检查和固定容量完整重跑：Flash非MTP70000 blocks，Decision16000 blocks。没有修改模型算法、同步顺序或数值阈值。

## 环境、checkpoint与输入

- 8×NVIDIA H200，每卡143771MiB；UUID/拓扑已记录。驱动570.133.20，CUDA Toolkit12.8.93，x86_64双路Xeon Platinum8558，192 logical CPUs、约2TiB内存；本会话无额外CPU/内存cgroup硬上限。
- Python3.12.14、Torch2.10.0+cu128、CUDA Triton3.6.0、FLA/fla-core0.5.2、NumPy2.2.6、Transformers5.16.0；模型环境Ray2.58.0。完整146包依赖和40971个依赖文件的SHA256保留，公共依赖检查通过。
- 复制既有CUDA依赖到独立目录，保留用户原环境；两个venv共享同一依赖快照，去除复制件中的旧editable路由，显式设置工作树PYTHONPATH。父进程、MP/Ray子进程使用导入守卫和路径审计。没有安装或导入dlinfer、torch_npu、triton-ascend，没有使用Ascend requirements/A3脚本。
- 固定BF16激活；FP8模型使用已有官方FP8文件，没有现场量化或替换小模型。固定Python/NumPy/Torch/CUDA seed42，greedy、do_sample=False/top_k=1；max_batch8、session65536、prefill2048、DP1/EP1，GLM-5.3按FlashMLA要求block64，其余按记录的实际配置。首次错误block128被两边同样拒绝，失败另存。
- 每轮包括单请求、8请求批次、长短混批、同prompt新请求重放、长上下文和重放。生成模型长输入实际32603 token；Decision最长29264。prefix命中实测，生成长请求可命中32512 token，Decision可命中29184。

| checkpoint | 固定内容来源 | 完整权重规模 |
| --- | --- | --- |
| GLM-5.3-Flash | 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a | 62分片，328326771576 bytes |
| Sage-A1.5 | ee49fa7353820fff9e59b7e89c51a729e8359a61 | 15分片，71903645408 bytes |
| Agents-A1 | 604894d081728ba6616387a2711215b79a95cb03 | 14分片，70216369184 bytes |
| Intern-S2-Preview-397B-FP8 | b9b98067e5a0cea541b9e4a2739c0aa7a5ddb667 | 96分片，408283814880 bytes |
| GLM-5.3 | e0b07fd2751b42d5efa199cc02c2b271deadc516 | 141分片，755617140416 bytes |
| Intern-Decision-4B | 用户zskj-hub目录；19个运行文件与官方38c9dafbbef52dd04bae943c0c6ac918419adfd5相同 | 4分片，9078626736 bytes |

Decision实际路径是/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--internlm--Intern-Decision-4B。README和.gitattributes与下载版本不同，因此没有把整个用户目录冒称为那个git revision。所有使用文件均实际读取并记录SHA256，配对使用同一物理文件。

## 数值检查与单测

比较生成token ID、完整top-5 ID集合和每个返回值的二进制编码，不用文本相同替代数值检查。API浮点数保存FP32与Python FP64的hex编码；计算dtype保持原路径。例如S2非MTP原生BF16 log-softmax能逐bit复现保存分数，强制FP32的负控不匹配，未以改变精度制造一致。发现首个差异后保留此前一致历史，不沿已不同的生成历史比较后续分数。

| 检查 | 基线 | 固定PR | 结论 |
| --- | --- | --- | --- |
| 共同model-agent/加载/config | 298通过、1跳过 | 298通过、1跳过 | 原跳过来自只暴露一张GPU；随后双GPU output-wait文件两侧各7/7，补齐该项 |
| 新增RMSNormGated及调用者/CPU合同 | 使用旧FLA/工厂或独立参考，不因缺新API判失败 | 122通过、1跳过、1按禁令排除 | norm83/83含56真实CUDA/graph；调用者8/8；会导入模拟torch_npu的合同排除 |
| f64关键文件复核 | 共同测试已有对照 | 273通过 | model_agent182、norm83、调用者8；此前3个cache_config异常在固定HEAD不再失败 |
| 相关CUDA kernel原套件 | 123通过、24跳过、1失败 | 相同 | 同一GDN断言失败保留；24项缺参考包 |
| 独立卷积参考补跑 | 32/32 | 32/32 | 含原先24个跳过项；单测专用env加入官方causal-conv1d1.7.0，模型环境未变 |
| Decision协议/编码合同 | 281通过、9跳过 | 相同 | 权重路径确认后，真实checkpoint预处理文件两边各9/9 |
| 真实94层norm权重 | 冻结旧工厂 | 282 eager、564 graph重放逐字节相同 | Flash34层、Sage30层、Agents30层；不替代完整模型 |

共同GDN失败为test_fused_gated_delta_rule_transposed_state_indices[4-False-state_dtype0]：两边同在(121,0,7,58)，1048576个元素中1个超过原1e-3断言，max-abs0.00390625。未改断言或算法；这是本环境在未修改基线也有的失败，未进一步认定是内核还是参考实现的问题。

卷积参考只装在envs/unit-oracle。官方1.7.0源码固定cd81f0413cad2fc1e6f17e785ac39f59aae690cd，使用本机CUDA12.8和gcc/g++11构建；该解释器实际sys.path的147包依赖检查通过。模型envs/base、envs/pr及共享CUDA依赖保持原状。

Decision按真实score-only任务、temperature1.99241824、HF任务覆盖配置执行，既比较62符号及额外GT候选分数，也比较/systemone与/v1/decisions响应。图审计两边各236次捕获，其中140个含CUDA kernel，真实请求各1224次非空图重放；57960分数及全部响应字段与正常graph/prefix运行相同。[图证据](evidence/cuda-h200-20261116/decision-graph-proof.json)。该审计只保留图定义并读取节点，不改变图中计算，诊断运行不计性能。

## 差异定位：保留失败而非放宽验收

### 1. 初始独立Triton缓存

Sage/Agents/Flash初期可各自重复，但跨源码首token分数不同。Sage冻结了真实输入、位置、KV/GDN/卷积状态、逐层hidden与完整target logits；第0层gated norm的输入/gate/scale相同，输出有1个BF16元素不同。两套自动调优缓存分别选BT16/warps16和BT16/warps8。

| 源码/冻结输入 | 原基线缓存 | 原PR缓存 |
| --- | --- | --- |
| 原基线工厂 | 复现原基线输出 | 复现原PR输出 |
| PR标准norm | 复现原基线输出 | 复现原PR输出 |

基线整模型换用原PR缓存，也精确复现原PR全部1408 token/7040分数。差异跟随缓存而非源码。额外匹配profile使用原基线缓存的完整副本供双方使用，没有手选winner、改kernel或放宽阈值。原失败和原始缓存、IR/cubin均保留；主表的条件通过不能覆盖独立autotuning环境下的失败。

### 2. GLM-5.3长上下文

补充核对原始输出：8种配置 × 基线/PR各自自重复 × 长请求/重放，共32组长响应对照，32组首个输出位置的分数不等，其中28组后续生成token序列也不等。这是已发生的重复性失败，不只是对确定性的担忧。[逐组原始文件与哈希](evidence/cuda-h200-20261116/glm-long-token-repeat-audit.json)。

“固定实际顺序”是冻结输入后的诊断性干预：分别把真实运行中捕获的两份索引数组喂给原attention算子，验证能否重现各自原输出。它没有修复selector，没有改变正常引擎，也没有作为修复后完整模型通过的证据。

所有8个TP rank冻结了第二个长prefill chunk。输入/位置、第0层Q/K/V、有效index scores和所选索引集合相同，索引顺序不同，attention输出开始不同，随后hidden与target logits不同。未修改的默认selector在同一冻结输入上32次产生32种顺序，集合始终相同；分别固定两次实际产生的顺序，attention各重复3次稳定并精确复现对应输出。基线/PR均如此。默认原子输出偏移导致顺序变化；没有启用deterministic模式、排序后送入attention或改变计算来制造通过。

### 3. HTTP动态合批和排列

此处包含两种因素：四个HTTP请求可能实际合成4或3+1/1+3等批次；即使批量仍为4，请求在批内的行位置也可能不同。缓存固定不能消除这两种数值敏感性。

S2的out_proj是第0层GDN/线性注意力子层经过gated norm后的输出投影，不是最终词表lm_head。在TP8下，各rank先做自己的局部投影，再合成完整hidden维度的结果；首轮观察点位于整个模块之后，因而覆盖本地GEMM/量化路径与TP归约，当时没有分开记录归约前的局部输出。2026-11-17新增完整8卡冻结算子重放，已将本次反例定位到BF16 NCCL Ring归约：本地投影逐位对应，只归约已保存的局部输出即可复现差异；详见[进一步归因](cuda_determinism_followup_20261117.md)。

补充证据边界复核：原首差分析来自rank0。复核两源码×8rank的16组orderA/orderB记录，14组能够按相同首个decode步骤对齐；这些记录中out_proj对应请求的输入相同，移位行的模块输出不同。基线rank3、PR rank4的orderB记录history_lengths为168（目标首decode应为167），且缺少该轮prefill记录，已排除。这两组不能用其他rank代替，也不能仅凭这批记录宣称完成全8卡同一步的归约归因。[全rank复核及排除原因](evidence/cuda-h200-20261116/outproj-all-rank-input-audit.json)。后续独立控制使用完整orderA的8卡输入构造一致的行置换，没有补写旧缺失记录；它精确重现原有效A/B输出，进一步隔离了本地计算与归约。旧排除项仍有效，未插桩主矩阵原始结果保持不变。

Flash/S2的真实HTTP C4在基线自身也不重复，故不能把这些时序用于合格的迁移性能通过。完整输入、首个数值差异与此前一致的生成历史，以及64份实际TP rank首prefill tensor均保存。

- Flash实际批量4与3的共同请求，首差在第0层fg_a_proj稠密投影；相同批量4的对照，逐层输出和target logits相同。
- S2实际批量1与4的首差在第0层MoE gate。用真实冻结输入/权重，在基线和PR源码各重放三次：两种形状的实际输出全部精确复现；Flash对应投影同样通过，共24次重放。不是以“BF16误差”忽略差异。
- S2另有首次decode才出现的变化。控制同一进程的两个批内排列，并保存真实CUDA graph逐层观察buffer、target logits及KV/GDN/卷积状态：相同历史、位置与逻辑状态下，两边均从第0层linear_attn.out_proj输出开始不同；未移动的那一行保持相等。首轮只定位到包含本地投影和TP归约的模块边界；后续[完整8卡冻结重放](cuda_determinism_followup_20261117.md)在两份源码均复现BF16 RING_LL归约的6736/16384个差异元素，并用逐步舍入参考精确解释全部输出。相同排列下跨源码的全部逐层输出和target logits相同；观察器原生dtype重算的前两个top-k分数与原未插桩HTTP记录相同。

decode图观察器增加了复制节点，只用于定位；原始性能与正常精度样本不含这些节点。PR观察进程完成8个响应、16个rank tensor后退出卡住，记录并仅清理本任务进程树；没有据此声称正常PR退出回归或正常退出通过。正常主矩阵和性能进程均另有完成记录。该插桩退出异常的归因仍未验证。

### 4. 首次cache miss与后续prefix hit

单请求和批请求的新请求重放均有记录。额外逐项审计发现：Sage/Agents/Flash的prefix长请求miss→hit，以及Decision的部分长评分miss→hit，分数会变化，两源码表现一致；S2本轮这些重放相同。GLM长重放亦失败。首次miss和后续hit的计算/cache条件不同，本报告没有把跨命中状态等价当成已证明事实；其完整数值等价性仍属基线限制。迁移对照的每个对应事件仍使用相同命中条件并逐bit比较。[重放审计](evidence/cuda-h200-20261116/intra-process-replay-audit.json)。

## 性能方法与结果

本节以[2026-11-18充分预热复测](cuda_warm_performance_20261118.md)为准。原先较大的TTFT增幅已由新结果取代，不继续作为当前问题清单。

新一轮固定原源码、依赖、checkpoint、GPU与cache配置，完成170个选定模型进程、每格五对AB/BA。满足数值前提的31个短/评分配置及GLM的两项32K保序迁移控制均通过既定持续回退判据；16组RMSNormGated算子配置也未触发回退。GLM的保序开/关成本与迁移对照分开记录。

动态HTTP C4的GLM-5.3-Flash、Intern-S2及本轮新增GLM，在两份源码各自重放时仍出现token/完整分数差异；这些配置不签署严格性能通过。没有改变算法、同步、collective精度或阈值来消除该限制。

预热覆盖引擎所有解码图桶与真实prefill/tuning形状；加载、编译、预热和数值探针不计入稳态性能。完整数值、区间、显存采样、首差及复现脚本均在新报告中。旧性能表只保留于[历史归档](old/cuda_performance_before_rerun_20261118.md)。

## Ray 独立 CPU 对照

无Torch/NPU的独立Python3.12.14 / Ray2.48.0 / NumPy2.2.6环境，参考报告ray_deep_probe_v2.py逐字节一致，SHA256为870843b45f906a203ef8639c15d041bed878ea3489e1895889de737b8c8583b5。保留submit→sleep12ms→join→free顺序、默认亲和性；没有照搬CPU18规避。

两次全新进程各6000次（put/direct各3000），put最大join45.123/40.834ms，direct1.444/2.078ms，无>50ms join。保存投递、确认、反序列化、event-loop、进程maps、线程schedstat及softirq时间线，未与GPU模型性能计时混跑。[数值摘要](evidence/cuda-h200-20261116/ray-summary-audit.json)。A3参考Python是3.12.13；有限x86窗口未复现不能认定ARM根因或宣布Ray问题解决。后面的GPU插桩退出异常是另一个未归因现象，不与此CPU对照混为一谈。

## 复现与证据

本机根目录：/home/daixu/data/pr5134-cuda-20261116（真实路径/nvme1/home/daixu/pr5134-cuda-20261116），工作区入口/home/daixu/GLM5.3-flash/pr5134-cuda-20261116。

- worktrees/base、worktrees/pr-f64固定源码；envs/base、envs/pr使用共享cuda-deps；所有验证命令、环境、helper版本SHA与对应源码快照在logs/*.command.json和evidence/script-snapshots。
- checkpoint-inventory及三组checkpoint SHA清单、dependencies/dependency-files-sha256、source-sha256、GPU UUID/拓扑、有效cache容量和每进程导入记录均保留。
- 主数值profile：Sage/Agents为matched-f64；Flash非MTP为matched-f64-tp8-isolated、MTP3为matched-f64-tp8；S2为matched-f64-tp8；Decision为matched-f64-decision-isolated。使用primary-summary/primary-matrix选择合格文件，不把资源作废轮次或TP4补充混进统计。
- scripts/model_case.py、decision_pipeline.py、model_queue.py、compare.py、perf_queue.py及各诊断/审计脚本保留完整参数；重跑使用新的label/output，启动器不会覆盖既有结果。相同条件包括实际cache blocks与保存的Triton缓存，不能只复制cache fraction。
- 原始逐token/候选score bits、性能每轮值、失败日志、CPU/图/数学oracle、CUDA冻结tensor、编译/调优产物均保留。匹配Triton缓存完整快照含12354文件、463866373 bytes；没有手工改调优winner。

发布的[分卷/文件SHA清单](evidence/cuda-h200-20261116/manifest.json)列出所有压缩分卷和逐文件hash。按manifest顺序拼接evidence.tar.gz.partNNN，再解压。大型tensor保留本机，manifest明确标published=false、绝对路径、字节数及SHA256；没有声称数百GB tensor已上传。完整报告只在fork文档分支。

最终归档为8个分卷、202615739 bytes；拼接SHA256为cac444d390240f9bac61de9d5ddd19a87c1986eadb7a402095ba73cf6720607a。26722个已发布文件按归档实际字节全部复核，119个大型本机文件约555.75GiB另有SHA；无不稳定JSON被跳过。[归档验证](evidence/cuda-h200-20261116/verification.json)、[结束时checkpoint复核](evidence/cuda-h200-20261116/checkpoint-final-audit.json)可直接审阅。末次核对时GPU恢复为GPU0约116MiB、其余约4MiB的初始空闲状态。

## 尚未验证的范围

未认证未获权重的Intern-S2非Preview FP8版本、用户示例DP4/EP8/router生产链路、跨机、真实外网HTTP、视觉模型GPU前向、生产高并发/长时间soak、MTP HTTP性能、target主干独立性能或标准能力榜单。Decision的图为现有piecewise图，部分操作按原设计eager执行。HTTP C4的动态合批/排列不能取得本轮要求的普遍逐位重复性；prefix miss→hit完整数值等价性未被证明。GDN共享单测失败及仅插桩退出卡住的原因保持开放。out_proj此反例已由后续冻结实验定位到BF16 TP Ring归约，但修复方案的完整模型正确性、性能与其他通信配置仍未验证。

这些限制不被改写成迁移通过，也没有为其放宽阈值、启用不同算法或混入基线修复。
