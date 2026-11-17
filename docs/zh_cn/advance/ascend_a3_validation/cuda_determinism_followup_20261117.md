# H200 CUDA确定性问题：原子索引、动态合批与TP归约的进一步归因

本轮固定基线442097e83799c2e85df9d6bbce35c7c79ea05dfe与PR f64ab5061153b5c5b63721cf2d8d6f1b41b95104。继续使用原H200、CUDA依赖与冻结输入，不更新main/PR版本，不修改实现源码。这里是[固定版本回归报告](cuda_regression_20261116.md)的诊断补充，不能替代完整模型或生产配置的修复验收。

## 本轮结论

| 现象 | 进一步验证的直接机制 | 同一物理输入布局重放 | 本轮未做的事 |
| --- | --- | --- | --- |
| GLM-5.3长上下文 | sparse_index_topk用整数原子计数器分配输出槽位，线程到达顺序改变索引排列，继而改变attention浮点归约分块 | 默认路径仍不重复 | 未启用新配置宣称完整模型已修复 |
| Flash/S2 prefill随实际batch大小变化 | cuBLASLt为不同矩阵行数选择Split-K/非Split-K GEMM，求和组织不同 | 各固定形状可重复 | 未把FP32/禁止减精度控制当作通用修复 |
| S2 decode随请求行排列变化 | 本地FP8投影逐位对应；BF16 NCCL RING_LL的分块归约顺序随元素位置变化 | 固定布局可重复 | 未把FP32归约控制写入原实现，未做该方案完整模型性能验收 |
| 四请求被分成3+1 | HTTP请求分别准备，RequestManager只取当前队列快照；scheduler处理当前就绪项，不等凑满目标数 | 取决于准备/入队时点 | 未更改排队策略强行凑batch，也未把诊断时间当服务性能 |

需要区分“同样输入重跑不确定”和“换batch布局后数值变化”。固定seed、软件版本或Triton缓存都不自动提供跨batch布局的逐位不变性。下述三类数值问题在固定基线与PR中均存在，不是本轮确认的迁移新增回归。

## 1. GLM：具体是哪一个atomic，以及为什么短输入可能不暴露

实际路径是lmdeploy/pytorch/kernels/cuda/sparse_index_topk.py中的TileLang byte-radix选择器。它有两类整数原子操作：

1. 直方图计数：对int32桶加1。在当前长度范围没有溢出，计数是精确整数；这不是本例的浮点累加误差来源。
2. 输出槽位预留：默认unordered分支用AtomicAddRet取旧计数作为slot，再写Out[row, slot]。线程/warp抢到slot的先后没有固定次序，所以集合相同也会排列不同。

生成的CUDA源码中，默认路径实际包含AtomicAddRet((&shared_state[0]),1)，并将其返回值用作Out的下标。生成源码随证据保存，不只引用Python层描述。

当真实KV长度不超过top_k时，代码直接按逻辑位置0、1、2……输出，不走这条输出预留路径。此GLM配置top_k为2048；第一个2048-token chunk尚不触发，第二个chunk超过阈值后开始暴露。这与上轮实际首差位置一致。这里的top-k是注意力KV位置选择，不是生成采样top_k=1。

使用原始冻结scores、Q、KV和真实长度，本轮在两份固定源码各执行下列检查，每格32次：

| 原生选择器模式 | eager：排列/attention输出种数 | graph：排列/attention输出种数 | 所选集合 |
| --- | --- | --- | --- |
| 默认deterministic=False | 32 / 32 | 32 / 32 | 始终与原始集合相同 |
| 独立诊断控制deterministic=True | 1 / 1 | 1 / 1 | 仍与原始集合相同 |

本次进一步调用了源码已有的稳定发射分支，由GPU重新计算索引，并不是把某一份固定索引数组反复传入。它是一个真实算子级控制；正常模型配置和原始失败记录未改变，也没有完成启用此分支后的完整GLM模型验证。

另做全等分数的合成tie控制：默认路径仍有多个输出顺序，PR侧eager的32次还出现2种有效索引集合；稳定分支始终固定。该控制说明cutoff ties也需要明确tie-break，但不能把合成case的集合变化冒称为上轮真实GLM输入的现象：真实冻结输入观察到的集合始终相同。

## 2. 为什么四个HTTP请求会成为3+1

本轮服务配置是DP1/EP1，不存在router把这些请求分发到多个实例的情况。外部“并发4”只表示四个独立请求同时在途，不表示服务端一次收到一个不可分割的四请求对象。

客户端threading.Barrier仅对齐发起时点，之后仍分别经过连接、HTTP解析、模板/tokenizer、RPC和引擎队列。当前RequestManager.get_all_requests先读取qsize，再取走当时已在队列中的项目；没有等待第4个的逻辑。PrefillScheduler从当前waiting列表选取，直到队列空或容量/预算达到上限。max_batch_size=8是上限，不是要求攒满8或4的门槛。

原未插桩日志perf-agents-http-pair0-pr-f64.log有如下记录：

| 时刻（2026-11-16，Asia/Shanghai） | 事件 |
| --- | --- |
| 20:13:47.960–.961 | 前3个请求的前端准备日志 |
| 20:13:47.963 | 一次取到3个ADD_MESSAGE；第4个前端准备日志也出现在该毫秒附近 |
| 20:13:48.053 | 下一次取到剩下1个ADD_MESSAGE |

这些短请求每个167 token，四个共668，低于2048-token预算；max_batch8也足够。决定性的观察是请求未作为同一个队列快照进入处理，而不是模型主动认为3比4更优。毫秒日志不足以把最后一个请求的延迟精确归因给网络、tokenizer或RPC中的某一段，因此没有作这样的断言。

另外直接调用未改的RequestManager.get_all_requests做CPU到达时序控制：四个预先就绪，得到[4]；先就绪三个、第四个稍后入队，得到[3,1]，两次均如此。此项仅验证真实队列代码的行为，不是GPU/HTTP性能测量。

源码中确有每次notify一个发送者、sleep100微秒的循环，但EngineInstance先send_async(ADD_MESSAGE)，随后async_recv(wait_main=True)才进入该等待。因此它位于入队后的响应等待环节，不能把它当成这次首次3+1的入队根因。

### 组成4是否更快

如果四个请求已经就绪，一批通常可减少一次prefill启动并提高矩阵利用率；但若第4个尚未就绪，等它会给前三个增加队列等待，若它不再到来，硬等还可能无期限拖延。连续batching通常在吞吐和TTFT之间取舍，不是每一步都最大化batch大小。3+1描述的是prefill阶段，后续decode仍可能合成4。

本例还有一个独立冷形状成本：167/334/501行是MoE tune_hint1，668行是hint2。第一次真正合4可能触发约491ms在线调优，而3+1不跨此阈值；这不是batch4的稳态计算天然更慢。前一报告已经把完整预热后的五对对照单列。本轮未修改等待策略，也没有给出“总等满4更快”的通用结论。

## 3. Flash/S2 prefill：矩阵形状真的改变了GEMM路径

用上轮真实冻结投影输入和相同权重，在单GPU上重放不含TP归约的局部计算。对应逻辑请求的输入行逐位相同，原生结果也逐位匹配之前捕获的实际投影/路由输出。Profiler显示：

| 算子 | 小行数 | 大行数 | 实际CUDA路径变化 |
| --- | --- | --- | --- |
| Flash第0层fg_a_proj，BF16权重256×4096 | M=510（3×170） | M=680（4×170） | 小形状出现Split-K GEMM及splitKreduce_kernel；大形状是非Split-K单GEMM |
| S2第0层MoE gate，BF16权重512×4096 | M=167 | M=668（4×167） | 同样从Split-K切换到非Split-K |

这里的M是送入矩阵乘的token行数，不只是HTTP连接数。Split-K把K方向求和分段后再合并；在有限精度下，结果不必与另一种分块/求和组织逐bit相同。

Flash共同请求每170×256个投影元素中有20个数值不同，最大绝对差0.0078125；S2共同请求167×512个原始router-logit元素中29个不同，最大绝对差0.0625。后面的softmax/路由和深层计算会继续使用这些不同值。本轮不把这种差异忽略为通过，也没有由此推断业务准确率下降多少。

反例控制同样保留：关闭allow_bf16_reduced_precision_reduction，差异仍在；TF32关闭、FP32计算后转回BF16，在这些共同请求中仍各留1个差异元素。因此不能简单声称“关一个减精度选项”或“转FP32”就能保证任意batch不变。具体GEMM指令级逐项累加轨迹没有采集；已直接观察到kernel/分解变化并复现了实际结果。

## 4. S2 decode out_proj：已从模块边界缩到BF16 TP AllReduce

上轮14组可对齐rank记录证明模块入口相同、移位行出口不同；另2份错步记录仍保持排除。这次没有拼接那两份记录，而是用完整orderA的8卡真实输入和对应真实FP8权重，构造一个明确的全8卡排列P(orderA)，建立独立的受控重放。

每rank实际局部输入为1×4×1024，权重4096×1024、FP8 E4M3、scale_fmt=ue8m0，输出BF16。调用未改的CUDA provider，实际选择DeepGemmLinearBlockedF8Impl。分别保存通信前的局部投影与通信后的结果；TP=8、原8块H200，未指定新的NCCL算法/协议。

| 检查 | 基线与PR的全部8个rank |
| --- | --- |
| 本地量化+FP8 GEMM结果与行置换交换顺序 | 逐位相等 |
| 同一局部输入重复 | 逐位相等 |
| 原生BF16 AllReduce后，再按逻辑请求对齐 | 6736/16384个元素不同，max-abs0.0078125 |
| 只置换已保存的局部投影，再做AllReduce | 精确复现同一差异，排除局部GEMM为该反例来源 |
| 固定布局重复AllReduce | 逐位相等 |
| eager/真实CUDA graph重放 | 两者均精确匹配原始A输出和有效B输出，差异模式相同 |
| 仅用于诊断的FP32归约后转BF16 | 本fixture中置换后相等，且与FP64求和后转BF16的参考相同 |

Profiler记录的实际通信kernel是ncclDevKernel_AllReduce_Sum_bf16_RING_LL，NCCL2.27.5；本次调用16384个BF16元素、8rank、1个CUDA grid block。日志中ring为0→1→…→7。

### 为什么请求换行会改变Ring归约结果

NCCL的[runRing源码（v2.27.5-1）](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/all_reduce.h)按元素位置分块，chunk c依次累加rank c+1、c+2……c。此调用可以由8个2048元素块精确解释：每个请求有4096个输出元素，所以占两个块；请求换行后，同一逻辑元素可能进入不同块，所经历的环形相加顺序随之改变。

以CPU做明确的“每一步加法后舍入回BF16”参考，按上述源码顺序预测两种排列：**A与B的全部16384个结果均逐bit匹配实际NCCL输出**，两份源码都成立。不是只观察到相关性或笼统归咎于精度。

一个真实标量例子：同一请求从行3移到行0，其8个rank的局部数值完全没变；相加顺序从7,0,1,2,3,4,5,6变成1,2,3,4,5,6,7,0，结果分别为0.003570556640625和0.0035552978515625。逐步舍入序列完整保存。

因此，这个S2 decode反例确实来自TP通信中的BF16归约顺序。它与GLM的随机槽位排列不同：固定同一布局时Ring重放稳定；请求行置换改变的是确定的分块/归约次序。没有证据表明这是通信丢数据，也不能把它泛化成所有TP/NCCL配置都必然有相同问题。

## 5. sparse_index_topk有几种实现、六个模型如何调用

本节按固定442097/f64源码和实际checkpoint配置核对，未更新实现版本。严格按函数名，CUDA目录只有一个公开的sparse_index_topk入口，其TileLang byte-radix实现有两个编译模式：默认deterministic=False用原子槽位无序发射；deterministic=True通过前缀计数按逻辑索引升序发射，同分截断时优先较小逻辑索引。k=512、2048是两个K特化，不是两种独立算法；真实长度≤K时直接输出有效索引也是同一实现内的短路径。

相关但不同的实现还包括Triton bitonic_topk（NSA在TileLang不可用或K不支持时的回退，以及V4的形状/类型回退）和sparse_dcp_global_topk（DCP多分片候选归并）。DCP归并不能修复上游已经不稳定的本地候选选择；本轮DCP=1，不将其列为已验证修复。

| 本次实际模型 | 注意力位置选择路径 | 是否使用共享sparse_index_topk |
| --- | --- | --- |
| GLM-5.3，GlmMoeDsaForCausalLM | GlmMoeDsaIndexer → IndexerTopKFP8 → CUDA NSA | 是，K=2048，调用没有传deterministic，默认False |
| GLM-5.3-Flash，Glm5NextForConditionalGeneration | KPool-4分组选择，短prefill有直接索引路径 | 是，group_topk=2048/4=512，明确deterministic=True |
| Sage-A1.5 | Qwen3.5 MoE的GDN+全注意力 | 不走该注意力位置选择器 |
| Agents-A1 | Qwen3.5 MoE的GDN+全注意力 | 不走该注意力位置选择器 |
| Intern-S2-Preview-397B-FP8 | 映射到Qwen3.5 MoE路径 | 不走该注意力位置选择器 |
| Intern-Decision-4B | Qwen3.5的GDN+全注意力 | 不走该注意力位置选择器 |

因此，不是只有GLM-5.3使用这个函数；Flash也使用，但已有稳定发射。仓库范围还有DeepSeek V3.2/V4调用者。本表的“top-k”专指注意力KV位置，不能与MoE专家路由或生成采样top-k混为一谈。

源码依据：[默认NSA调用](https://github.com/InternLM/lmdeploy/blob/442097e83799c2e85df9d6bbce35c7c79ea05dfe/lmdeploy/pytorch/backends/cuda/nsa.py#L472)、[KPool稳定调用](https://github.com/InternLM/lmdeploy/blob/442097e83799c2e85df9d6bbce35c7c79ea05dfe/lmdeploy/pytorch/backends/cuda/kpool.py#L353)、[选择器两种发射模式](https://github.com/InternLM/lmdeploy/blob/442097e83799c2e85df9d6bbce35c7c79ea05dfe/lmdeploy/pytorch/kernels/cuda/sparse_index_topk.py#L136)。

若目标是可重复，应明确约定“同一选中集合按逻辑索引升序，同分优先较小逻辑索引”。这比只对最终数组排序更完整：默认路径在cutoff ties时可能已选出不同集合，事后排序无法补救。也不需要为了保序改成按score降序的完整排序；当前deterministic=True已经实现稳定压缩。接口sorted=True表示另一种score排序要求，当前函数会拒绝它。稳定索引只修复这一处机制，仍依赖相同scores，并不能自动解决上游GEMM或下游TP的跨batch数值变化。

## 6. S2是否同一布局重跑就随机、其他模型TP是否免疫

本次NCCL RING_LL反例由缓冲区分块决定rank累加顺序，不是哪个rank先到就先加的原子竞争。固定rank映射、内核、元素数与排列时，冻结重放重复相同；移动请求行后，同一逻辑元素落入另一chunk，使用另一条确定的循环累加顺序。前节的CPU参考精确匹配全部输出，支持的是这一具体机制，不是对任意NCCL配置的全面确定性保证。

两个服务请求即使prompt、seed相同，动态合批的行位置也可能不同，因此未修改的旧服务连续调用仍可能出现不同结果。基线与PR均已复现，且固定排列后96项跨源码tensor比较相同；无需这次迁移改动即可触发。一般而言，改变主机时序的代码可能改变合批从而暴露已有敏感性，但没有证据将本例归类为PR新增的归约算法回归。

其他模型没有天然豁免，必须检查实际通信路径和测试覆盖：

| 模型/条件 | 本轮证据与边界 |
| --- | --- |
| Flash，TP8、DP1/EP1 | 固定基线已在KDA/稀疏注意力输出投影设置tp_reduce_dtype=float32，MLP共享投影及MoE最终TP归约也有FP32路径；与S2本次BF16路径不同。Flash仍有此前定位的局部GEMM形状敏感性，不能称完全batch invariant |
| S2，TP8 | 该GDN out_proj没有上述升精度覆盖，原生BF16 Ring反例已实测定位 |
| GLM-5.3，TP8 | 长上下文已经在selector先出现首差，不能由此证明后续所有TP归约都不受排列影响 |
| Sage/Agents/Decision，TP1 | 本轮没有跨卡TP归约，不能把单卡通过延伸成其TP2/4/8也已验证；共享模型路径在多卡下仍需单独检查 |

Flash原有配置见[输出投影](https://github.com/InternLM/lmdeploy/blob/442097e83799c2e85df9d6bbce35c7c79ea05dfe/lmdeploy/pytorch/models/glm5_next.py#L712)与[MoE最终归约](https://github.com/InternLM/lmdeploy/blob/442097e83799c2e85df9d6bbce35c7c79ea05dfe/lmdeploy/pytorch/models/glm5_next.py#L547)。这里没有宣称FP32相加满足结合律或能保证全部输入逐位不变；当前冻结S2控制有效，通用保证还需要稳定的分解和求和顺序。

## 7. 其他引擎的TP归约精度：传输类型与累加类型要分开

本节是官方仓库固定SHA的源码审计，没有运行这些引擎做完整模型对比。vLLM固定61c3895fc937b4ffc2a59abe188959de1e459c8a，SGLang固定f534dbaeb2d07c67951d905c628d2888e1ca1673，TensorRT-LLM固定62a91924391ee7fbd66cd000179629d0a90db1ff；不把结论无条件套到其他版本、硬件或通信后端。

| 引擎及被核对路径 | 输入/输出与跨卡读取的数据类型 | 内部累加 |
| --- | --- | --- |
| vLLM常见CUDA CustomAllreduce | FP16/BF16输入仍以原类型在共享/P2P缓冲区读取，最后存回原类型 | packed_t的A是float数组；全部rank输入转FP32累加后再转回 |
| SGLang常见CUDA CustomAllreduce | 同样保留FP16/BF16输入与输出 | 同样使用FP32累加器 |
| TensorRT-LLM当前ar_fusion oneshot/twoshot | 保留输入DType的通信缓冲区与输出 | 当前源码默认FP32累加；ALL_REDUCE_FUSION_KERNEL_ACC_FP32=0可退回原dtype累加 |
| 三者各自的NCCL路径 | 按输入tensor dtype调用ncclAllReduce，没有统一、自动升FP32的承诺 | 由选中的NCCL内核决定；不能把“引擎支持FP32自定义归约”套到所有NCCL调用 |

依据：[vLLM累加器与packed_reduce](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/csrc/custom_collective_common.cuh#L83)、[SGLang对应实现](https://github.com/sgl-project/sglang/blob/f534dbaeb2d07c67951d905c628d2888e1ca1673/python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh#L69)、[TensorRT-LLM累加实现](https://github.com/NVIDIA/TensorRT-LLM/blob/62a91924391ee7fbd66cd000179629d0a90db1ff/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu#L378)及[默认开关](https://github.com/NVIDIA/TensorRT-LLM/blob/62a91924391ee7fbd66cd000179629d0a90db1ff/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu#L723)。NCCL dtype传递分别见[vLLM](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/vllm/distributed/device_communicators/pynccl.py)、[SGLang](https://github.com/sgl-project/sglang/blob/f534dbaeb2d07c67951d905c628d2888e1ca1673/python/sglang/srt/distributed/device_communicators/pynccl.py#L161)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/62a91924391ee7fbd66cd000179629d0a90db1ff/cpp/tensorrt_llm/thop/allreduceOp.cpp#L433)。

FP32寄存器累加并不要求先把整个BF16通信tensor转为FP32：自定义kernel可以读取各rank的BF16贡献，在本地FP32寄存器中完成求和，再只舍入一次。它与前节诊断用的“整块转FP32再调用NCCL”有不同的通信量、工作区和性能取舍，不能把两者都说成payload翻倍。实际速度和拓扑适用性必须测量。

精度提升本身也不是绝对保序：所查1stage路径对所有元素按固定rank顺序求和；SGLang所查2stage路径仍按分块负责rank旋转读取顺序，并且按消息大小切换实现。本轮重新读取所引用的vLLM文件，2stage的输入指针为ptrs[i]=_dp->ptrs[i]，已使用绝对rank顺序，轮转的是stage2分发使用的tmps；此前将vLLM这一实现也描述成owner-first求和的文本予以更正。要承诺整个模型严格跨batch一致性，仍须约束其他通信后端选择及上游GEMM等，不能仅凭某一归约路径的float累加器推断。[vLLM重新核对的代码](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/csrc/custom_all_reduce.cuh#L44)。

六个实际config哈希、11份LMDeploy源文件哈希、10份官方外部源码的固定URL和本地快照哈希见[调用链与外部引擎审计清单](evidence/cuda-engine-paths-20261117/review-manifest.json)。外部源码快照留在本机artifacts/engine-collective-review-20261117；不把这次静态审计计作其他引擎性能或模型验收。

## 8. 其他引擎是否默认让稀疏Top-K保序

不能把一个引擎名称或deterministic参数概括成“所有稀疏Top-K均按逻辑索引保序”。本次继续按前节相同的vLLM/SGLang固定SHA读取CUDA DSA路径，只做源码分析，没有运行其他引擎的完整模型。

- vLLM的SparseIndexerTopk默认decode选择链为cooperative→persistent→per_row；前两者的内核都有用atomicAdd预留输出位置的路径。cooperative将这些顺序写入最终row_output，persistent也直接用原子槽位写最终输出，不能保证完整数组按逻辑索引保序。auto路径还会随工作形状选择实现。依据：[默认分派](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/vllm/model_executor/layers/indexer_topk.py#L295)、[cooperative输出](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/csrc/cooperative_topk.cuh#L419)、[persistent输出](https://github.com/vllm-project/vllm/blob/61c3895fc937b4ffc2a59abe188959de1e459c8a/csrc/persistent_topk.cuh#L575)。
- SGLang当前默认启用Top-K v2，SGLANG_TOPK_V2_MODE默认1：保证选择值的正确性，但cutoff同分可选任意候选。mode2/deterministic=True加强到同分按索引选择、选中集合固定。关键是源码把mode2的保证写成selected set相同，其select_smallest_indices仍明确不保证发射顺序，并使用原子输出槽位，因此不能把mode2等同LMDeploy的完整逻辑索引升序发射。依据：[默认配置](https://github.com/sgl-project/sglang/blob/f534dbaeb2d07c67951d905c628d2888e1ca1673/python/sglang/srt/environ.py#L1663)、[模式契约](https://github.com/sgl-project/sglang/blob/f534dbaeb2d07c67951d905c628d2888e1ca1673/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh#L74)、[确定性tie选择仍可无序发射](https://github.com/sgl-project/sglang/blob/f534dbaeb2d07c67951d905c628d2888e1ca1673/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh#L396)。

这不等于已实测这些引擎的完整模型必然不重复：还要检查具体模型、后端、prefill/decode分派及attention消费者。本文结论限定为“所查默认selector不提供全路径逻辑索引保序保证”。未来LMDeploy修复的验收应分别检查选中集合和发射顺序，不能只检查排序后的集合相同。8份官方源码的固定URL及本地快照哈希见[selector源码审计清单](evidence/cuda-selector-engines-20261117/review-manifest.json)。

## 9. 固定布局可重复与一阶段/两阶段归约

需要先区别TP输入通道分片与AllReduce的通信分块。实际S2配置为64个value heads、每head128维，即out_proj总输入8192维；TP8后每rank输入1024维。每rank权重形状为4096×1024，因此本地投影产出完整4096维的部分和，随后才在8rank间求和。若代入16个请求，形状依次为逻辑输入[16,8192]、每rank输入[16,1024]、每rank局部输出[16,4096]。这是按实际配置推导的16请求示例，不是新增GPU实验；原冻结实验实际为4请求，每rank输出[1,4,4096]。

NCCL的AllReduce接收缓冲区地址、元素总数及dtype，不接收二维矩阵的列维度。本次runRing通过gridOffset、elemOffset和chunk*chunkCount寻址一维连续片段；不能假定它对每个请求独立按列切1/8。[固定NCCL版本的runRing源码](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/all_reduce.h#L13)。对用户提出的[16,8192]通信张量，若仅作“单channel、一次连续等分8块”的简化示例，每块16384元素，会包含两整行，而不是16行各1024列。实际较大消息还受channel、chunk大小及循环轮次影响，未测该形状时不能断言其真实划分正是这8块。

如果实际实现确实逐行按列分块，而且同一列始终使用相同的累加表达式，那么单纯换行确实不会由分块导致数值变化；部分具体形状也可能恰好具有这样的重复分块模式。已确认的S2反例是另一种情况：4×4096输出展平后，8个2048元素块能精确解释实际NCCL结果，每个请求占两个块，换行会改变对应块。固定环拓扑不意味着所有块从同一rank开始归约；不同块沿同一环的不同起点累加，形成循环移位的浮点求和序列。

当前S2反例中，“固定”有两个层次：对一个物理缓冲区分块，Ring使用确定的rank循环顺序；对同一个逻辑请求跨不同batch行位置，其分块可能变化，累加顺序就不再相同。于是迁移前A与迁移后A可以逐位相同，A自身也可重复；但同一请求移动到排列B后，A/B仍可不同。迁移前后均能复现这个限制，与原子竞争导致同一实际输入重复也不稳定的GLM selector不是同一机制。

严格的修复要求是所有逻辑元素都使用同一条FP32求和序列，例如rank0→1→…→7，最后只转回一次BF16；顺序不能随消息大小、行位置、分块owner改变。固定NCCL_ALGO=Ring只固定算法家族，并没有赋予这个更强的数值约束。

这里的一阶段/两阶段指vLLM式自定义AllReduce组织：

| 方案 | 归约与分发过程 | 典型取舍 |
| --- | --- | --- |
| 一阶段 | 每张卡读取所有rank的局部贡献，对整个输出独立计算完整和；每卡都按0→7顺序 | 阶段少；同一元素的完整求和在各卡重复，跨卡读取较多，常用于小消息 |
| 两阶段 | 先将输出划成8块，每卡负责一块并读取所有rank该块的贡献做归约；再all-gather各卡已经求和的块，让每卡拿到完整输出 | 每个元素只需做一份完整归约，减少重复读取/计算，但增加分发和同步阶段，较适合大消息 |

两阶段本身并不要求改变求和顺序。所查SGLang两阶段代码用(owner+i)%TP组织输入指针，所以owner变化会轮转加法顺序；本轮重读的vLLM两阶段代码已经让输入指针按绝对rank读取，其分发临时缓冲仍可轮转。未来实现应分别核对输入读取和结果分发，不把后者的轮转误当成前者；让每个owner按逻辑rank0→7求和，最后all-gather只复制最终BF16结果。NCCL Ring也可分为reduce-scatter/all-gather两个通信阶段，但逐hop处理和舍入方式不等于上述P2P自定义内核。这项外部源码描述更正不改变原S2冻结输入、NCCL源码哈希或实际CUDA结果。

BF16传输、FP32内部累加只消除这处归约敏感性；若进入归约的各rank局部GEMM结果已随batch形状变化，完整模型仍可能不一致，必须独立记录。

用户要求的后续实现任务已保存为[独立修复Prompt](cuda_determinism_independent_task_prompt_20261117.md)。本次只编写后续任务与核对源码，未创建实现分支、未启动修复。该Prompt要求从固定基线独立分支实施，在单机8卡验证batch大小/排列及性能，并保持与迁移PR隔离。

## 修复方向与尚未验证部分

- GLM的稳定索引发射分支在本轮真实冻结算子输入上有效；尚未把它启用为正常模型策略，未完成完整模型精度/性能回归。
- S2此TP归约反例的FP32累加控制有效；其通信payload相对BF16增加一倍，这不等于实际延迟一定翻倍。完整模型吞吐、显存、融合路径及其他层的效果尚未验证，未把它混入迁移PR。
- Flash/S2局部GEMM的形状选择是另一条问题线，单改TP归约不能同时解决它。严格batch不变性需要进一步控制或设计GEMM分解，并检查性能代价。
- 在线凑batch需要有上限的等待策略及吞吐/TTFT实测，不能无条件等满4；本轮只分析当前行为。
- 旧错步采集文件、插桩退出问题、prefix miss→hit等价性等限制没有被这次局部实验覆盖。主矩阵和原始失败不被重新标为无条件通过。

## 原始证据与复现

新增证据位于[evidence/cuda-determinism-20261117](evidence/cuda-determinism-20261117/index.json)：两份源码各8rank的投影/归约张量、逐项比较、eager/graph结果、CUDA profiler trace、NCCL日志、CPU逐步舍入参考、选择器生成CUDA、原始正常HTTP日志摘录、队列控制、命令/环境与脚本。NCCL源码快照附其许可证。大模型权重和旧大tensor不重复上传，原始输入路径、size/mtime/SHA引用已保存。

发布前已逐文件复核SHA256，并在CPU重新加载两份源码各8rank的输出文件：96项跨源码tensor比较全部按原dtype逐bit一致，见[证据复核记录](evidence/cuda-determinism-20261117/verification.json)。这项复核确认两份源码得到相同的局部结果、相同的原生归约结果和相同的诊断控制结果，并不抹去各自跨排列的6736个差异元素。

数据仅来自固定442097/f64，没有更新实现分支、修改默认算法、安装NPU依赖或改动原始验收阈值。反事实控制均明确标注，GPU实验退出后恢复原有空闲状态。
