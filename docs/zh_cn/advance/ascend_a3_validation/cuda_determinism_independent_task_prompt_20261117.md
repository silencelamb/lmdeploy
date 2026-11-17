# 后续独立任务 Prompt：GLM-5.3 稳定索引与 S2 确定性 TP 归约

进展说明（2026-11-17）：GLM-5.3已完成[仅开启现有稳定选择器的8卡完整模型控制](glm53_stable_selector_fullmodel_20261117.md)，32次新进程、8组匹配配置对照通过。已观察到的跨prefix miss→hit生成候选top-5数值差异只作记录，不再作为本次排查/修复要求；同缓存状态的新进程和迁移配对仍须零容差。以下后续任务可复用这些证据，但仍需最终生产策略接入后的验证及性能评估。Intern-S2的确定性归约尚未在本轮实现。

请在这台单机 8×NVIDIA H200 上，独立修复并验证两项已确认的 CUDA 基线确定性问题：

1. GLM-5.3 的 DSA `sparse_index_topk` 启用稳定选中集合与逻辑索引保序。
2. Intern-S2-Preview-397B-FP8 的 GDN `out_proj` 参考 vLLM CustomAllreduce，实现 BF16 通信缓冲区、FP32 内部累加、与 batch 大小及请求行位置无关的固定 TP rank 求和顺序。

这是迁移前已经存在的问题修复。必须在用户 fork 的独立实现分支和工作树完成，不修改、不合并到 Ascend 迁移 PR #5134，也不要自动合并任何 PR。直接完成代码、测试与性能分析，无需逐步确认；只有无法从机器、已有记录或代码中解决的关键缺失信息才询问，并继续独立工作。

## 1. 固定版本、环境与现有证据

- 先读取本机适用的 AGENTS.md，检查已有目录、环境、GPU占用和git状态，保留用户已有数据与任务。
- 未修改对照固定为 `442097e83799c2e85df9d6bbce35c7c79ea05dfe`。从此SHA创建独立修复分支/工作树，例如 `fix/cuda-deterministic-selector-tp`；另建未修改对照工作树。若同名分支已有工作，先检查，禁止覆盖。不要更新或追随移动的main。
- `f64ab5061153b5c5b63721cf2d8d6f1b41b95104` 是上轮迁移PR的固定参考，不是本任务要更新的分支。核实实际git SHA与工作树导入路径，防止editable install串环境。
- 同一批8卡、同一驱动及CUDA/Python/PyTorch/Triton/TileLang/FLA/NCCL依赖、同一checkpoint文件；保存版本、环境变量、实际通信后端及源码/权重哈希。禁止安装或导入dlinfer、torch_npu、triton-ascend，禁止使用Ascend requirements或A3亲和性启动规避。
- 保存并匹配编译、Triton/TileLang及autotune缓存。新增kernel使用与源码SHA绑定的独立构建产物，不能误加载旧二进制。原有算子的缓存winner必须可核对，不能以独立调优造成的差异冒充源码回归。
- 旧记录根目录：`/home/daixu/data/pr5134-cuda-20261116`。主要冻结证据在 `followup-20261117/`，旧大tensor及权重引用在 `results/` 和已发布manifest内。原始失败与证据不可覆盖，所有新实验使用新label。
- 阅读[主回归报告](cuda_regression_20261116.md)及[确定性机制报告](cuda_determinism_followup_20261117.md)，包括错步rank记录的排除说明。旧的两份错步orderB记录不能用于拼接全8卡参考；使用后来完整8卡受控重放的有效证据。

实际模型路径，执行时核实快照与文件哈希：

- GLM-5.3：`/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516`
- 用户已批准的S2版本：`/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--internlm--Intern-S2-Preview-397B-FP8/snapshots/b9b98067e5a0cea541b9e4a2739c0aa7a5ddb667`

两个完整模型均在单机8卡、TP8测试。不换小模型、不擅自改量化，也不把Preview结果冒称为非Preview版本通过。

## 2. GLM-5.3：稳定集合和索引输出顺序

- 当前CUDA入口为 `lmdeploy/pytorch/kernels/cuda/sparse_index_topk.py`，已有 `deterministic=True` 分支。GLM-5.3经 `backends/cuda/nsa.py` 默认调用无序分支；Flash的KPool调用已显式保序。优先复用现有稳定实现，在模型/后端合适的配置边界接入，避免无依据改变所有共享调用者。
- 明确语义：按score选出最大的K项；cutoff同分优先较小逻辑KV索引；选中索引按逻辑索引升序发射；有效长度以外填充值不变。`sorted=True` 不是现有接口支持的保序开关。不能只排序旧结果而忽略同分时选中集合也可能不同。
- 用独立数学参考验证选择集合、同分规则、索引顺序和padding。覆盖K=512/2048、长度K−1/K/K+1、尾部不满tile、非连续行stride、全相等及部分同分、正负零、有效长度在graph重放时变化；其他输入特殊值按既有接口契约处理，不静默改语义。
- 用真实GLM冻结scores/Q/KV重放默认与稳定路径，保留默认失败。稳定路径要求重复、eager/graph、batch拆并与行置换后，选中集合、索引序列和对应attention输出逐位一致（对应逻辑输入相同）。
- 检查AR与MTP target/draft相关调用是否都传递所需策略；不要只修prefill、遗漏decode或验证阶段。验证共享Flash/其他受影响调用者接口，不能因改默认参数意外改变无关模型行为。

## 3. S2：BF16通信、FP32累加、所有元素相同的rank顺序

- 已知首差是第0层 `model.language_model.layers.0.linear_attn.out_proj`。各rank本地量化/FP8 GEMM对应请求的结果相同，原生BF16 NCCL RING_LL在行置换后产生6736/16384个不同元素。固定布局重放稳定；不同块循环rank起点不同。先在当前环境复现这个冻结反例，不把它解释成随机网络到达或浮点atomicAdd竞争。
- 区分8192输入通道的TP切分和投影后4096维部分和的AllReduce。后者按一维通信缓冲区寻址，不默认对每行独立按列切分。对每个实际batch/shape记录channel、chunk和归约路径，不能假定任意消息都恰好分成总元素数/8，也不能从某个恰好对齐的形状推广全部排列都一致。
- 参考固定vLLM源码 `61c3895fc937b4ffc2a59abe188959de1e459c8a` 的 `csrc/custom_collective_common.cuh` 与 `csrc/custom_all_reduce.cuh`，记录引用和必要许可证。输入/P2P通信缓冲及最终输出保持BF16，使用FP32累加器；所有逻辑元素始终按TP逻辑rank `0,1,...,7` 进行相同的FP32加法序列，最后只转回一次BF16。
- 先清点LMDeploy已有communicator及optimized all-reduce能力，优先复用或扩展适当的通信接口；检查blocked-FP8投影当前如何选择归约路径，避免在模型层重复实现整套通信栈。策略接入范围要明确，不能无依据改变所有模型的默认路径。
- rank顺序必须与请求行位置、消息大小、分块owner和warp/CTA调度无关。先做简单可验证的一阶段方案也可以；如支持两阶段，reduce-scatter各owner仍按全局固定rank顺序累加，all-gather只复制最终结果。不能直接照搬owner-first轮转的两阶段路径，然后声称已经保序。
- 固定NCCL算法名/协议、固定seed、设置同步或CPU亲和性，都不等于固定每个逻辑元素的数值归约树。不能以此替代实现。
- 显式把整个通信tensor转FP32再调用NCCL仅保留为诊断对照，不能冒称已经实现BF16通信。也不能假定自定义BF16通信＋FP32累加一定更快；实际流量、工作区、同步和延迟分别测量。
- 处理连续性/对齐、非整块尾部、TP组及rank映射、多个并发缓冲、CUDA graph capture/replay、buffer生命周期、跨卡可见性与同步。必要时核对生成代码，避免编译器改变所约定的加法序列。不要通过插入全局同步制造正确性。
- 明确支持边界和fallback。严格确定性路径遇到不支持条件，不能静默退回会破坏承诺的BF16 Ring；给出可验证的确定性fallback或清晰报错。其他dtype、TP大小和拓扑保留原语义，不把未测范围标为支持。
- oracle分开记录：一份按固定rank顺序逐步FP32舍入、最终转BF16的精确参考；另用FP64参考评估数值误差。FP32并不满足结合律，不能要求它对所有数值都等于FP64后转BF16，也不能把“用FP32”本身当作固定顺序证明。

## 4. 单机8卡正确性验收

- 先验证两个冻结算子，再验证两个完整模型。非MTP在前，受支持的MTP3在后。Python/NumPy/Torch/CUDA种子固定，greedy `do_sample=False, top_k=1`。同dtype、TP、checkpoint、输入token序列、输出长度、sampling、cache块数及命中状态。
- batch大小至少1/2/3/4/8；固定同一批逻辑请求，覆盖正序、逆序、循环置换、batch4的全部24种排列及batch8的固定随机排列。覆盖一次合4与3+1/1+3等拆分、短请求与长短混批、同prompt新请求重放。实际服务端合批/排列要记录，不能仅凭客户端并发参数推断。
- eager/graph、prefix off/on分别覆盖。在prefix off或相同cache命中状态下检验batch不变性；已记录的GLM miss→hit生成top-5差异仅作现象记录，不扩大为本次新的排查要求，不混淆比较条件。长上下文至少覆盖2048选择阈值两侧及代表性的约32K，按真实token数记录。
- 原版和修复版每个关键配置各至少两次全新进程自重复。比较同一逻辑请求在相同历史、位置下的完整生成token、完整top-k候选集合及每项分数原dtype二进制，零容差。固定rank的collective输出也要所有rank分别核对，不能只看rank0。
- 验收对象是修复后的自重复、跨batch/排列一致性和数学语义。旧版自身不稳定，因此不能把“修复版必须逐位复现旧版任意一次无序输出”当作验收标准；旧版用于保留反例、量化改变与性能对照。
- 首差即保存此前一致的token历史、draft IDs、位置、KV/GDN/卷积状态、各rank通信前局部结果、通信后结果、target logits及逐层hidden。比较冻结的同一历史，禁止沿已经分叉的生成历史继续归因。
- 已知S2/Flash另有本地GEMM随batch形状变化的问题。两项修复不自动保证完整模型batch invariant：若仍在GEMM或其他模块首差，精确定位、单列剩余问题与下一步方案；不能把算子通过替代完整模型通过，不能悄悄放宽阈值或修改调度强行凑批。

## 5. 性能代价

- 正确性验收后，对相同checkpoint、实际输入/输出token数、并发、cache命中条件做性能对照。预热所有实际形状和调优bucket，保留冷启动成本记录；不能重演遗漏batch4调优形状的问题。
- 预先固定回退判据：至少5对AB/BA交错；配对中位不利变化>5%、至少4/5轮>5%、配对bootstrap95%区间排除0时进入定位。保存每轮值、样本量与噪声，不只报平均值。
- 分别报告selector、TP collective、含本地GEMM的out_proj、固定轨迹target主干、pipeline与HTTP端到端；记录吞吐/TPS、TTFT、TPOT、峰值显存/通信工作区。禁止将插桩或冻结诊断耗时混入正常推理计时。
- 旧版与修复版若生成轨迹或MTP接受率不同，端到端可比性限制必须单列，并补充同一冻结轨迹的target/collective对照；不可比轮次不能用来宣布性能合格。完整模型正确性仍失败的配置，性能只能作为诊断数据，不能标“验收通过”。

## 6. 交付

- 实现仅在fork独立分支，提供可审阅提交和针对性测试；不更新迁移PR #5134，不自动合并。
- 报告继续维护在fork的 `docs/ascend-a3-validation-20261114` 文档分支，完整报告和原始证据不混入实现提交。
- 保存全部环境、命令、源码与checkpoint哈希、实际kernel/通信路径、编译和调优产物、逐token/score、失败首差、每轮性能数据与复现脚本。保留用户原目录/环境和旧失败记录。
- 最终分别给出“GLM稳定选择器”“S2确定性归约”“各完整模型跨batch一致性”“性能代价”矩阵，区分通过、回归、其他基线问题、未验证。说明哪些结论来自真实8卡运行、哪些来自静态分析或诊断控制。
