# PyTorch Engine 的模型专用执行接口

这个接口把一次 **target 模型 forward** 交给一个可选的 `SpecializedExecutor`。合作方可以生成一个常驻核、一组由 stream 编排的核，或者复用模型中的若干层。LMDeploy 负责调度、位置/页表、KV 和命名状态分配、checkpoint、LM head、采样与发布 token。用户通过 `PytorchEngineConfig.specialized_executor` 或 `serve api_server --specialized-executor 模块:类` 显式选择本机已安装的实现。默认值为 `None`。

## 为什么选择这个边界

SGLang #44353 将版本、模型能力、权重观察、TP 一致性和 Graph 覆盖纳入框架，厂商单独交付计算实现。vLLM #61867 的 Kimi K3 在模型权重就绪后建立 prepared providers，随 KV 分配绑定布局，`plan` 检查 capture、batch 和 verify 条件，`run` 编排多层。后者以 GB300 TP8/TP16 为目标，显式拒绝 EP。这些工作提供了可比较的对象生命周期，各自的加速比不适用于 H200 DP8/EP8。[SGLang spec](https://github.com/sgl-project/sglang/blob/ec7ab9d0c21ae8e1d146379f5c148df9edae2176/python/sglang/srt/monokernels/spec.py) · [SGLang 安装与覆盖校验](https://github.com/sgl-project/sglang/blob/ec7ab9d0c21ae8e1d146379f5c148df9edae2176/python/sglang/srt/monokernels/runtime.py) · [vLLM plan/run](https://github.com/vllm-project/vllm/blob/642c18d64c4cb6db3ded8fbe1786718cd4005fcb/vllm/models/kimi_k3/nvidia/specialized_decode/runtime/decode.py) · [vLLM 的模型与 KV 生命周期](https://github.com/vllm-project/vllm/blob/642c18d64c4cb6db3ded8fbe1786718cd4005fcb/vllm/models/kimi_k3/nvidia/model_state.py)

LMDeploy 已有两处需要复用的边界。`model_forward` 构建 `StepContext`，挂接 block/state cache，恢复 checkpoint 并准备模型输入，forward 后整理隐藏态、cursor 和 checkpoint。CUDA runner 已将模型对象与用于捕获的 callable 分开，负责 fixed-address buffers 和 replay。V1 在这两处之间选择 callable，保留标准模型及其模块。显式的类路径可把生成物作为独立包部署，省去了扫描 entry points、管理全局 registry 和永久替换 `model.forward` 的复杂度。[ModelAgent](../../../lmdeploy/pytorch/engine/model_agent/agent.py) · [CUDA runner](../../../lmdeploy/pytorch/backends/cuda/graph_runner/runner.py) · [Full Graph 捕获和重绑定](../../../lmdeploy/pytorch/backends/cuda/graph_runner/full_graph.py)

## 生成物和引擎各自承诺什么

| 时点                             | 引擎提供                                                                                                                                                                           | 生成物负责                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 静态探测 `describe(ctx)`         | `BuildContext(model, model_config, cache_config, backend_config, dist, device, num_spec_tokens=0)`。原模型已完成 HF 加载、sharding、量化模块的 `update_weights` 和 post-load hooks | 核对模型族/层图、真实张量 shape/dtype/stride、量化及 scale、设备 ISA、拓扑、权重版本约束和可执行模式。用无副作用的 `tuple[DecodePlan, ...]` 声明覆盖，空元组表示当前部署走默认路径                                                                                                                                                                   |
| 准备 `__init__(ctx)`             | 确认各 rank 的实现路径、API 版本、生成物标识和有序计划一致，再建立实例。初始准备位于 KV 容量测量之前                                                                               | 打包本 rank 的派生权重，建立本地 scratch、加载 native 库并使资源对 forward stream 可见。构造阶段只做本地工作。全部 worker 成功后，运行时调用 `prepare_distributed(ctx)`。完整 R11 在该阶段绑定既有 EP 组、建立对称内存并按相同顺序交换所有层的副本。跨 rank 计算可使用已建立的引擎进程组。需要新的 collective 分配协议时须有全组有序初始化和失败收敛 |
| 选路                             | `DecodePlan(name, batch_size, max_kv_len, graph_safe, num_spec_tokens=0)`。正整数 batch，含当前 token 的 KV 上界或 `None`，按声明顺序取第一个匹配项                                | 每个计划覆盖所有满足条件的行长度、槽号和值。仅支持等长历史、某种页排列、特定下界或其他 V1 无法表达的 kernel 应声明空覆盖或在静态绑定时证明条件恒成立                                                                                                                                                                                                 |
| 执行 `run(plan, **model_inputs)` | 同一 target 的 `prepare_inputs_for_generation` 产物，当前 `StepContext`、执行流和既有后端。Graph 路径传稳定缓冲绑定后的参数                                                        | 保留 forward 的输出张量或字典、形状、dtype 和可选字段，以及对权威 KV/状态的全部写入。辅助 stream 进入前等待生产者，返回前把依赖接回调用流。保持需要的 `record_stream`/存活期                                                                                                                                                                         |
| 结束 `close()`                   | 先退休 graph 并排空设备上的使用者，随后丢弃 executor。权重刷新重新从完成处理的参数构造，sleep/wake 按当前资源重新准备                                                              | 释放设备 module、派生权重和 workspace。不得将同地址的原地权重更新、同名缓存或重用 arena 视为同一代数据                                                                                                                                                                                                                                               |

Python 接口只依赖框架模型、配置与设备对象。CUDA driver handle、cubin、stream ID 和 Triton 均在 H200 示例中。V1 由 NVIDIA CUDA runner 实现 dispatch，A3 继续使用 `AscendGraphRunner`。NPU 方可实现同样的 `describe/run/close` 并在自己的 runner 绑定相应 eager/graph callable。需要在该设备上验证地址稳定性、stream/event、通信和缓存写回。API 版本由 `SpecializedExecutor.api_version` 声明，生成器用 `artifact_id` 标识实际程序、权重布局与通信协议，原生库 ABI 和编译器/驱动兼容性随生成包交付。[接口类型](../../../lmdeploy/pytorch/specialized_execution.py) · [A3 runner](../../../lmdeploy/pytorch/backends/ascend/graph_runner.py)

## 单步数据与动态负载

1. 支持普通 AR decode（每行一个 query），以及显式注册的 `deepseek_mtp` target verify（每行 `num_spec_tokens + 1` 个 query）。只有部署深度与计划一致、真实 token 数等于 B×Q、KV 上界满足时才能进入。Dummy、chunk、vision、microbatch、LoRA、MemDecode、ReplaySSM 和其他投机算法沿用已有路径。Prefill/混合 phase 由已有 eager/PCG 处理。
2. DP 的每个 worker 根据 host 元数据产生一个整数。0 表示默认，否则为有序计划的 1-based 下标。`_prepare_dp_v1` 在已有的 CPU all-gather 末尾附一个整数。全组一致、全局 decode、未启用本步 microbatch 才把结果写进 `ModelInputs.specialized_plan`。一张卡出现空行、另一种 bucket 或长度超限，全组同退。没有新增每步 collective 或设备到主机拷贝。单 DP 的本地选择要求参与普通 TP 执行的 ranks 接收同样的逻辑输入。当前实物 provider 仅声明 DP8/EP8/TP1。
3. runtime 每步重评，带修改的 `ModelInputs.clone` / `step` 清除上一步的选择。DP 缺少本步协商结果就走标准执行。计划的 B 是该 rank **真实**请求数，Full Graph 还校验捕获 bucket 与计划 B 正好相等。B7 填充到 B8、一个 rank 有 8 个非 decode token、MTP 的八个 verify 行，均不能借用 B8 AR 计划。
4. 权威数据包括 `input_ids`/`position_ids`、`attn_metadata` 的 q/kv lengths、query offsets、block table，以及模型准备好的 `past_key_values` 和 `state_ids`。运行时长度、物理页号、请求顺序、状态 input/output ID 可变。真实值由 device buffer 或单步参数提供。生成器不能将 capture 时的一次输入值、某个 data pointer 或当前行号固化为请求身份。
5. V1 采用引擎原生 cache ABI。GLM 的 BF16 latent、FP8 pooled key + FP32 scale、BF16 卷积历史、FP32 recurrent state 和两种 KPool tail 均有指定所有者。模型在这些缓冲上原地读写。引擎决定保存、恢复、复用和对外可见的时机。额外 scratch 可由 provider 持有，KV/state 视图仅限当前 forward 或引擎管理的图。跨步私有 shadow cache、新布局的迁移和事务提交协议尚未开放。

## CUDA Graph 与故障语义

图缓存保留 LMDeploy 原有的 batch、phase、microbatch、query 宽度和模型 extra key。只有选中 graph-safe 的计划才追加其不可变 `DecodePlan`。相同 B 的普通图、不同计划的专用图可以并存。Graph runner 使用同一个专用 callable 进行真实 warmup 和 capture，用普通模型的 fill/update/output hooks 重绑元数据、裁剪输出。首次 miss 返回 warmup 结果，避免同一个真实 step 的 SSM 写回重复执行。

启动 warmup 先准备标准图，再对配置的精确 bucket、每个 graph-safe 计划用全组保留的 dummy slots 预捕获。声明 graph-safe 即承诺这种合成 warmup 也合法，能处理无效状态索引，并在捕获内不做 JIT 初始化、host sync、资源 rendezvous 或缓存内容相关的 Python 分支。真实 dummy 请求仍回到标准图。关闭 warmup、资源刷新或新增模型 extra key 时沿用 runner 的按需捕获。普通 Full Graph 模式遇到 eager-only 计划使用标准图。PCG prefill 不接入此接口。

覆盖未命中在计算前决定。专用 callable 一旦开始，对 KV、recurrent state 或 mailbox 可能已有写入。异常直接传播，不能重跑旧 forward。启动时导入/版本/计划错误和组员准备失败会交换状态，再结束本次初始化。进程突然死亡、native 死锁及异步设备 fault 依赖 worker/process-group 的监控与重启。V1 没有请求级自动回滚。模型日志列出部署计划和含计划的图 key，DEBUG 可看到实际专用 eager 执行或 replay。

## MTP target verify 与提交边界

生成物可声明 `DecodePlan('verify-b2-d3', batch_size=2, graph_safe=True, num_spec_tokens=3)`。B 是真实请求数，Q=4 包含由上一轮确定的输入和三枚 draft token。AR B8 与 MTP B2×Q4 有相同的扁平行数，但使用不同的计划、图和状态语义。运行时过滤非本部署深度的计划，八卡对本步选择达成一致，再调用 target。请求的历史长度可以各不相同。当前计划要求每个请求有相同的候选宽度。

target 返回四个位置的 hidden 以及模型要求的辅助输出。使用引擎给定的 block table、每行的 input state ID 和按位置排列的 output state IDs 写入各候选的 KV、卷积、recurrent 和 KPool 状态。原有 rejection sampler 决定每行接受的连续草稿数 a。引擎将 history 前移 1+a，将 cursor 指向本步第 1+a 个输入后的状态。被拒绝的尾部物理内容可以覆盖，下一轮只能读取逻辑上保留的前缀。全部三个草稿命中时，第四个输出为 target bonus。普通路径也可以从同一组权威缓冲继续。私有跨步状态和自定义提交协议不在本版契约内。[引擎步进](../../../lmdeploy/pytorch/model_inputs.py) · [拒绝采样](../../../lmdeploy/pytorch/spec_decode/reject_sampler.py)

draft 由 LMDeploy 的 proposer 负责，有独立权重、cache、graph 和 rank 映射。当前生成物只接 target，不能把 draft 的通信组或状态槽当作 target 的资源。完整 R11 只注册 AR 计划。已有 B2×Q4 生成物只复用其中无状态的 34 个 KDA 输出投影。[MTP 实测与示例](glm53_specialized_mtp_deployment.md) · [SpecModelAgent](../../../lmdeploy/pytorch/spec_decode/spec_agent.py)

## 首版交付与实测入口

最小示例 `R11Projection` 替换 34 个 KDA 层的输出投影。PTX 取自 R11 发布物，清除了三行尾部空白。两个源文件在本机生成逐字节相同的 cubin，模板可单独在 sm_90a 上构建。Python pack、三段归并和 module 的释放在示例包内。已有模型继续计算 KDA 递推、KPool、MLA、mHC、dense FFN 和 MoE。选用 #5186 时，DeepGEMM/FlashInfer 的权重与通信仍由那个后端管理。完整 R11 的副本专家协议、设备自回环与 12.34909 ms 历史结果分别保留在审计文档。

8×H200 对照包括真实 checkpoint、短/长上下文、全组 B8 → 单 rank B7 → 全组 B8、请求行重排、两种图、eager、原生 kernel 计数和逐字节全词表与最终物理状态比较。编译、部署、数值和具体计时见合作方交付入口。[合作方指南及实测](../../../examples/specialized_execution/README.md) · [冻结 R11 技术审计](glm53_monokernel_audit.md)

完整 `FullR11` provider 已将模型的 45 层执行经同一接口接入。普通 target forward 仍可服务不匹配的 rank/batch。完整路径的原始口径复现、预捕获、动态回退、跨运行环境的 prefill 定位及真实 Serving profile 单独记录。[完整 R11 实测](glm53_full_r11_results.md) · [原生交付包与启动命令](../../../examples/specialized_execution/FULL_R11.md)
