# MTP target logits 与 Ray RPC 停顿定位

日期：2026-11-16，Asia/Shanghai；原始命令、日志时间保留 UTC。

**接续结论：MTP 已定位到 Ascend GDN 循环状态写回的自动多缓冲编译路径，并在独立研究分支完成修复验证；Ray 的约半秒停顿已追到本机 CPU 18 的周期性 softirq 占用，已验证进程 CPU 亲和性规避。具体内核回调仍未确认。**

修复：[dfa3049f](https://github.com/silencelamb/lmdeploy/commit/dfa3049f1575305b7564d6f15c48022661bd902a)，回归夹具补强：[d2d933da](https://github.com/silencelamb/lmdeploy/commit/d2d933dad7c6eb55b26ad5b83667c2077e23ca6c)，分支 `research/ascend-mtp-gdn-state-20261116`；未改动原生迁移 PR。详细实验、限制和复现见本文末尾[接续调查](#接续调查固定状态重放与主机调度定位)。

以下至“接续调查”之前保留首轮记录。当时确认了贪婪验证、target logits 分歧及真实 HTTP 的约 484 ms 停顿；其“尚未完成”条目描述首轮状态，最新完成状态以后续章节为准。

## 固定版本与范围

- 当前诊断源码：[5f1b1b3c](https://github.com/silencelamb/lmdeploy/commit/5f1b1b3c12b1c198813f65275f778c7ad292fb02)，实际完整版本以[命令元数据](evidence/mtp-ray-20261116/logs/mtp3-greedy-runtime-r0.json)为准。
- 历史迁移精度比较仍是 LMDeploy [fb5de87a](https://github.com/silencelamb/lmdeploy/commit/fb5de87a45df9999a19a569ed9441df8edfa6975) + dlinfer [3b909634](https://github.com/silencelamb/dlinfer/commit/3b9096341a41e9d067104a39d914e47ca2d7ec50)，对比 native [6421cefe](https://github.com/silencelamb/lmdeploy/commit/6421cefea94edd350e419f556158639556357a27)。不将本轮当前 PR 的重复运行改称旧、新迁移对照。
- 当前 PR 与开发分支未改动；诊断脚本全部位于独立验证目录。原生进程与 Ray worker 继续启用禁止发现/导入 dlinfer 的 guard。
- A3，aarch64，Python 3.12.13，Ray 2.48.0，Torch 2.10.0+cpu，torch_npu 2.10.0.post4；详见[环境与模型配置哈希](evidence/mtp-ray-20261116/inventory/environment.json)。
- [实验前约定](evidence/mtp-ray-20261116/ACCEPTANCE.md)、[完整证据索引与 SHA256](evidence/mtp-ray-20261116/index.json)。诊断会改变运行时序，不能把本轮时间当成无插桩性能验收。

## MTP：实际走的是贪婪分支

历史请求明确设置 `do_sample=False, top_k=1, temperature=1.0, random_seed=42`。
`temperature=1.0` 不会覆盖 `do_sample=False/top_k=1` 的贪婪策略。

当前调用路径为：

1. `SamplingInputs` 汇总得到 `max_top_k=1`；全部请求贪婪时，逐请求 `top_k` tensor 可以省略为 `None`，不代表丢失策略。
2. `RejectionSampler.forward` 直接取 `target_logits.argmax`，进入后端 `forward_greedy`。
3. 按顺序接受与 target argmax 一致的 draft 前缀；遇到首个不同 token，输出 target argmax 并丢弃后续 draft。若全部一致，增加一个 target bonus token。
4. 当前 Qwen3.5 MTP proposer 继承的 draft 选择本身也是 `argmax`。

这里有“接受/拒绝草稿”的逻辑，但没有随机决定接受概率。随机拒绝和 multinomial 路径在全贪婪策略下不会执行。
三个固定版本中的 `reject_sampler.py` 内容完全相同，SHA256 均为 `a64bf42a8d3488460420ad877ff18208e8d3bdc9423e02a720993feb17a30104`，见[源码核对](evidence/mtp-ray-20261116/inventory/source-sampling-path.json)。

### 运行时审计

Sage-A1.5、BF16、TP2、MTP3、prefix cache、graph decode；每个独立进程运行同一组单请求、8 请求、长短混合与重复请求，总计 21 条回复，每条固定生成 64 token。
请求明确贪婪，`random_seed=42`、`PYTHONHASHSEED=42`。

验证脚本把随机拒绝入口和 multinomial 入口改为立即报错；在实际 verifier 边界记录 CPU/NPU RNG 状态，
并以 CPU 上独立的“最长匹配前缀”规则核对 output、rejected count 和 last token。CPU oracle 使用实际记录的 target argmax，不是另一遍完整模型计算。

| 项目                           |   r0 |   r1 |
| ------------------------------ | ---: | ---: |
| verifier 调用次数              |  112 |  114 |
| 请求级验证行数                 |  448 |  449 |
| 实际 `max_top_k==1`            | 全部 | 全部 |
| verifier 前后 CPU/NPU RNG 不变 | 全部 | 全部 |
| 接受/拒绝输出符合 CPU oracle   | 全部 | 全部 |
| 随机路径触发                   |    0 |    0 |

两次共 897 个请求级验证结果均符合贪婪规则，仍只有 **15/21** 对回复的完整 token 序列相同。
这排除了本轮运行中“误走随机拒绝采样”及已检查的前缀接受规则错误；不能据此认证模型浮点计算、缓存回滚或所有 MTP 场景正确。
新审计没有额外重置所有 worker 的全局 RNG；历史全局固定种子重复实验及其基线自身不一致仍见[历史报告](migration_validation.md)。本轮主要证据是实际分支与 verifier RNG 未被消耗。

数据：[汇总与六处首次分歧](evidence/mtp-ray-20261116/results/mtp-runtime-summary.json)、[审计脚本](evidence/mtp-ray-20261116/scripts/mtp_audit.py.md)。

### 已找到进入 verifier 之前的分数变化

第一条单请求的第 17 个输出 token（零基索引 16）首次不同。此前 16 个输出 token、前五次验证的 draft IDs 和 target tensor 形状 `[1, 4, 248320]` 均一致；第五次验证在 block 的位置 0 发生分歧。

| 运行 | target 候选分数                      | 贪婪输出 |
| ---- | ------------------------------------ | -------- |
| r0   | token 728: 26.250；token 279: 26.125 | 728      |
| r1   | token 279、310、728 并列 26.125      | 279      |

r1 的第三个并列候选由返回的 logprobs 确认，原始 top-2 记录只有 279 和 310。
第五次调用的 draft 都是 `[728, 310, 12204]`。r0 接受首个 draft，r1 拒绝它并输出 target 的 279；两边都执行了正确的贪婪前缀规则。

这不是仅凭生成文本推测：首次分歧处记录到了 **target logits 的变化**。其上游原因仍可能涉及数值运算、模型状态、graph 或执行顺序；尚未锁定具体算子，不能简单写成“BF16 正常误差，已经解决”。
下一步应固定该请求和首个分歧位置，比较 target 隐状态、GDN/卷积状态、KV 与位置索引，逐层定位最早发生变化的位置；不能沿已经不同的文本继续比较后续 logits。

数据：[r0 单请求 target 跟踪](evidence/mtp-ray-20261116/results/first-single-r0-target-trace.json)、[r1 跟踪](evidence/mtp-ray-20261116/results/first-single-r1-target-trace.json)、[完整 r0](evidence/mtp-ray-20261116/results/mtp3-greedy-runtime-r0.json)、[完整 r1](evidence/mtp-ray-20261116/results/mtp3-greedy-runtime-r1.json)。

## Ray：在线请求中确实出现同类长停顿

真实 loopback HTTP API，Agents-A1、BF16、TP2、非 MTP、graph decode、无 prefix cache；
`session_len=8192`、`max_batch_size=8`、`max_prefill_token_num=256`、cache fraction 0.5。
C1/C4 表示 1/4 条并发请求，不是卡数。各预热一批，再测六批；30 条测量请求、另 5 条预热请求均返回 HTTP 200 和规定的 256 个输出 token。

保留原来的 `ray.put`、远程调用、`await asyncio.gather(*_prev_out)` 与释放顺序。
诊断仅增加输入标记、确认结果中的时间戳、主机日志及 10 ms 周期的事件循环心跳；输入标记在进入模型前移除。
这是真实 HTTP 链路上的诊断运行，不是无插桩吞吐基准，也没有进行迁移前后 HTTP 性能比较。

### 捕获到的时间线

第 5 个 C4 测量批次（`repeat=4`）的四条响应，在流式内容片段索引 116 附近同时出现 **483.79–484.04 ms** 间隔。
该间隔发生于稳定生成中段，关联 RPC item 3220。

| 阶段                                      |     rank 0 |     rank 1 |
| ----------------------------------------- | ---------: | ---------: |
| driver 开始 `ray.put` → worker 方法入口   | 469.306 ms | 468.345 ms |
| worker 方法入口 → 输入放入模型队列        |   0.094 ms |   0.061 ms |
| worker 方法结束 → driver 开始反序列化确认 |   1.739 ms |   1.646 ms |
| driver 反序列化确认                       |   0.041 ms |   0.066 ms |
| 反序列化完成 → gather 完成                |   0.131 ms |   1.194 ms |

对应 `ray.put` 本身约 **0.939 ms**，整个本次提交函数约 **2.201 ms**；下一次提交时等待上一组确认 **461.239 ms**。
该等待覆盖期间，driver 和两个 worker 均没有记录到超过 50 ms 的事件循环心跳迟到。
同一批次全部四条 SSE 流的约 484 ms 间隔与这一等待重合。

本轮共记录 3619 次 RPC join，其中一次等待超过 50 ms。**四条 SSE 慢间隔属于同一次共享停顿，不能算四个独立样本。**
SSE 内容片段也不保证与单个 token 一一对应，不能直接将间隔称为精确逐 token 设备时间。

`WorkerWrapperBase.forward_async` 只调用 `set_forward_inputs` 把输入放入队列；该 RPC 的确认不代表整个模型已经完成计算。
现在可以进一步把本次主要等待定位到 **worker Python 方法开始之前**，而非此方法内部执行、结果反序列化或回到 driver 事件循环的长等待。
还需要区分 Ray 任务投递、输入对象依赖就绪、raylet/core-worker 队列与调度；本轮日志不足以认定是哪一个 C++ 内部环节。

数据：[HTTP 原始记录](evidence/mtp-ray-20261116/results/http-agents-ray-r1.json)、[关联时间线与全部慢间隔](evidence/mtp-ray-20261116/results/http-agents-r1-summary.json)、[探针](evidence/mtp-ray-20261116/scripts/ray_trace.py.md)、[分析脚本](evidence/mtp-ray-20261116/scripts/analyze.py.md)。Ray 原始进程日志亦收录于证据目录。

### ARM 与 H200 的边界

本次实测平台只有 A3/aarch64，不能据此认定“ARM CPU 慢”或者“H200 一定不会发生”。
当前 CUDA/Ascend 的 Ray executor 共用输入提交代码，因此**在采用相同 Ray executor 的 H200 部署上，机制上也存在这条等待路径**；这只是代码路径判断，不是 H200 已复现。
换用其他 executor、Ray 版本、CPU 架构或系统配置时，需要重新验证。

[Ray 官方异步 actor 说明](https://docs.ray.io/en/latest/ray-core/actors/async_api.html)解释了 actor 的事件循环与协作调度模型；本报告的具体定位以安装的 Ray 2.48.0 和实测时间线为准，不把通用文档当作该停顿的根因证据。

另完成了[纯 Ray CPU 对照](evidence/mtp-ray-20261116/scripts/ray_cpu_control.py.md)：两个 async actor，8 KiB NumPy payload，交替 `put → direct → direct → put` 四个各 1500 步的 block；每次投递后 driver 等待 12 ms，再 join 前一次确认。driver 和两个 worker 均确认没有导入 Torch 或 torch_npu，没有模型或设备计算。

| 传参方式       | 双 worker join 数 | 最大确认等待 | 最大 dispatch → worker 入口 |
| -------------- | ----------------: | -----------: | --------------------------: |
| `ray.put/free` |              3000 |   546.999 ms |                  471.455 ms |
| 直接传参       |              3000 |   553.568 ms |                   28.080 ms |

`put` 的 6000 次 worker 投递中有 3 次超过 100 ms；直接传参没有超过 100 ms 的投递，但仍有长确认等待。
因此，**模型、Torch、NPU 计算不是在这台主机上触发 Ray 长等待的必要条件**，也不能宣称绕开 `ray.put` 就已经修复。
CPU 对照中的确认耗时尚未继续拆分为结果抵达与事件循环唤醒，不能把每个慢样本都强行归为与在线 item 3220 完全相同的内部触发器。
这一结果缩小了依赖范围，但依然没有把 Ray 版本/实现、ARM 主机与系统配置拆开；x86/H200 控制实验仍未完成。

数据：[CPU 对照摘要与慢样本](evidence/mtp-ray-20261116/results/ray-cpu-control-summary.json)、[完整 6000 步记录](evidence/mtp-ray-20261116/results/ray-cpu-control-r0.json)。
可在 H200 主机安装相同的 Python、Ray 2.48.0 与 NumPy 版本后直接运行该脚本，不需要加载模型或占用 GPU；具体版本见环境清单。

## 失败记录与尚未完成的验证

- HTTP r0 的诊断启动入口漏了 `__main__` 保护，在 multiprocessing spawn 阶段失败；修复验证脚本后 r1 完成。保留失败日志和原入口快照，不归因于 LMDeploy 生产代码。
- MTP 的当前证据覆盖 Sage-A1.5 MTP3；未宣称所有模型、混合采样策略、随机采样分布、各长度均通过。
- 尚未完成固定状态的逐层 MTP 数值定位；未将已有输出不一致算成按位通过。
- 未定位 Ray C++ 内部触发条件；未在 H200 或其他 CPU 架构完成同场景对照，也未认证性能问题已修复。
- 原来的 HTTP C4 迁移性能下降未被重新认定为回归；这里是当前源码的独立 RPC 机制调查，原始指标与比较范围见[此前性能报告](gated_norm_performance_20261116.md)。

## 复现与继续研究

本机根目录：`/root/workspace/a3-mtp-ray-investigation-20261116`。
`logs/*.json` 保存固定源码、工作目录、环境、完整命令和退出码；`scripts/run.py` 从 native checkout 启动独立环境。
再次运行请使用新 label、输出文件名和未占用端口，避免覆盖原始证据。

```bash
cd /root/workspace/a3-mtp-ray-investigation-20261116
python scripts/analyze.py --ray-label http-agents-r1 --http-file http-agents-ray-r1.json
```

MTP 的优先入口是第一条请求、第 17 个输出 token、第 5 个 verifier block；Ray 的优先入口是在线 trace item 3220。
后续应分别记录 worker 收到输入对象、反序列化开始/结束和 actor 方法调度的时间，并对照相同 Ray 版本下的 CPU-only、实际模型与 H200 环境。
所有后续修复保持独立研究分支，不把尚未确认的算法或调度变更混入原生迁移 PR。

## 接续调查：固定状态重放与主机调度定位

接续实验时间：2026-11-15 17:31–18:11 UTC，即 2026-11-16 Asia/Shanghai。
汇总：[机器可读结果](evidence/mtp-ray-deep-20261116/results/deep-investigation-summary-r2.json)；[证据索引](evidence/mtp-ray-deep-20261116/index.json)。

### 版本、隔离与判据

- 原始原生源码仍为 `5f1b1b3c12b1c198813f65275f778c7ad292fb02`；原生 PR checkout 保持干净。
- 计算核修复为 `dfa3049f1575305b7564d6f15c48022661bd902a`，仅修改 Ascend recurrent GDN 编译选项并新增硬件回归；最终研究分支 HEAD `d2d933dad7c6eb55b26ad5b83667c2077e23ca6c` 补强回归输入，计算核不变。开发分支、同事目录和当前迁移 PR 均未改动。
- 沿用交接的 native venv、模型与 Ray 2.48.0；未安装或导入 dlinfer。driver/worker 继续使用 import guard，环境审计确认包不可发现、未安装。详情：[环境](evidence/mtp-ray-deep-20261116/inventory/deep-environment-r2.json)。
- MTP 在 driver 和每个 worker 明确固定 Python、NumPy、Torch、NPU seed 42；请求保持 `do_sample=False, top_k=1, temperature=1, random_seed=42`。
- [实验前判据](evidence/mtp-ray-deep-20261116/ACCEPTANCE-deep-r1.md)：只比较相同输入历史；保存有效 state/cache、位置与层输出；不将同步拷贝探针当性能数据；Ray 保留全部 await、提交和释放顺序，>50 ms 仍作为慢样本。
- 下表另列 >300 ms，是对已观测半秒停顿的分类，不取代 >50 ms 判据。新 label 保留全部失败，未覆盖项不算通过。

### MTP：首次差异发生在 GDN 状态写回

对原生源码做两次独立、全局固定种子的单请求运行，Sage-A1.5、TP2、MTP3、BF16、graph、prefix 开启，生成 24 token。为容纳诊断克隆，将 cache fraction 从 0.95 调为 0.7；两边相同。随后无层快照的 21 响应验证恢复 0.95。

探针在设备上克隆层输出和缓存，graph capture 时把克隆也录入图，外层 forward 返回后才同步读回。保存两个 rank、40 层、前 8 次 target forward；attention KV 取活动请求首个 block，覆盖本例全部有效 token，GDN/卷积取活动 state ID 的全部槽。位置、state ID、cache length、KV block/index、输入 IDs 均保存。完整快照约 17 GiB，留在本机 `artifacts/mtp-deep-*`，哈希见[本机大文件清单](evidence/mtp-ray-deep-20261116/local-artifacts-manifest.json)。

原生 `layer-graph-r1/r2` 的时间顺序是：

1. 27-token prefill：两个 rank 的已记录层输出、卷积/GDN 状态及 KV **全部相同**。
2. 第一个 4-token target block：输入 `[90700,8340,25,198]`，位置 `[27,28,29,30]`、history length 27、state ID 1、初始缓存及所有已记录层输出仍相同；**写回后的 FP32 recurrent state 首次不同**，30 个 GDN 层均出现差异。卷积状态和 attention KV 此时没有差异。
3. 下一次 target forward：第 0 层在读取前述不同状态后开始产生输出差异，随后传播到后续层。
4. 两次前 16 个输出 token 相同，第 17 个再次分别为 728 / 279，复现交接入口。到 `step006` 输入 IDs 已不同，分析停止因果比较；未使用不同历史之后的分数解释根因。

数据：[原生逐层比较](evidence/mtp-ray-deep-20261116/results/mtp-compare-layer-r1-r2.json)。这里的第一处异常是 FP32 状态写回，不能归结为最终 BF16 logits 的普通舍入差。

### 固定输入的小复现与修复

将 GDN 拆成无模型、无 Ray、无 HCCL、无采样的单算子重放；每次调用前恢复完全相同的初始状态并同步完成后检查输出与所有写回槽。另从上述第 0 层快照提取真实 Q/K/V、投影 b/a、初始状态，并从固定 checkpoint 读取该 TP rank 的 A_log/dt_bias。修复前后这些输入、状态和参数的 SHA256 **逐项相同**。

| 固定输入重放 | 次数 | 输出不同于首次 | 状态不同于首次 | 观测最大状态差 |
| --- | ---: | ---: | ---: | ---: |
| 原始，合成输入，eager | 100 | 0 | 10 | 0.124178 |
| 原始，合成输入，graph | 100 | 0 | 99 | 0.770633 |
| 修复，合成输入，eager | 200 | 0 | 0 | 0 |
| 修复，合成输入，graph | 200 | 0 | 0 | 0 |
| 原始，真实第 0 层输入，graph | 200 | 0 | 199 | 0.456257 |
| 修复，同一真实输入，graph | 200 | 0 | 0 | 0 |

“不同于首次”是重复性计数，不等于逐次与真值比较的错误次数。为防止稳定地产生错误状态，还实现了独立 CPU FP32 recurrence，逐 token 计算每个 circular slot。预先规定状态 `atol=1e-5, rtol=1e-4`，BF16 输出 `atol=1e-3, rtol=1e-2`。真实输入原始 graph 首个快照的状态 max-abs 为 **0.456257**，不通过；修复后为 **1.90735e-6**，通过。两者当前输出均通过 BF16 输出检查。这解释了为什么只看本次输出会漏掉后续历史状态损坏。[真实输入 oracle](evidence/mtp-ray-deep-20261116/results/gdn-real-oracle-r0.json)

修复把 **circular-state 路径**的 `num_stages` 设为 1，其他路径保留 3。安装的 Triton Ascend compiler 明确把 `num_stages=1` 映射到 `--enable-auto-multi-buffer=False`；计算公式、dtype、接受/拒绝和回滚槽规则均未改变。[编译选项映射及源码哈希](evidence/mtp-ray-deep-20261116/inventory/triton-multibuffer-mapping-r1.json)

现有证据把问题定位到该 recurrent kernel 的自动多缓冲编译路径中的状态写回；尚未在编译器低层 IR 中指认具体缺失的依赖边或缓冲复用指令，不能把这个内部机理假说写成已证实的指令级结论。

验证结果：

- 新增硬件回归覆盖 query tokens 1/2/4 × eager/graph，共 **6 passed**；每项 32 次恢复初始状态，检查每个状态槽的 CPU oracle 和重复性。初始全部设备随机输入的一次旧核组合 pytest 未检出异常，保留为漏检观察；改为独立小复现的固定 CPU/device RNG 夹具并提前锁定实现模块来源后，原始源码在 2-token eager 项失败（max-abs 0.093989），修复源码六项通过。[补强后旧核失败](evidence/mtp-ray-deep-20261116/logs/gdn-pinned-original-r1.log)、[修复通过](evidence/mtp-ray-deep-20261116/logs/gdn-pinned-fixed-r1.log)
- 同一回归用明确断言源码路径的入口，在原始 `5f1b1b3c` 上失败，max-abs **0.588359**；研究源码通过。[失败日志](evidence/mtp-ray-deep-20261116/logs/gdn-hardware-original-direct-r0.log)、[修复日志](evidence/mtp-ray-deep-20261116/logs/gdn-hardware-fixed-direct-r0.log)
- 修复后两次单请求逐层快照，两个 rank、8 个 target forward 的已记录输入、位置、hidden、卷积/GDN 状态及 KV **全部逐元素相同**。[比较](evidence/mtp-ray-deep-20261116/results/mtp-compare-stage1-layer-r0-r1.json)
- 恢复原 0.95 cache fraction，Sage MTP3/TP2/graph/prefix、每条 64 token、交接的 21 响应序列独立跑两遍：**21/21 token IDs 完整相同，21/21 返回 top-5 raw logprobs 相同**。两遍各 112 次 verifier、447 条请求级验证，共 894 条，均为贪婪、CPU 接受规则一致、验证器 RNG 不变。这里只认证已保存的 top-5 分数，不声称检查了整轮所有词表 logits。
- 单算子 graph 计时，去掉前 10 次后的设备中位数为 **0.05215 → 0.05273 ms**，约 +1.1%；这是固定小形状微基准，不是整模型吞吐验收，且原实现的写回本身不正确。Ruff、AST 和 `git diff --check` 通过。

历史归属也已核对：旧 dlinfer `3b909634` 与原生 `5f1b1b3c` 的三个 recurrent GDN 函数 AST 完全相同，包含原来的 `num_stages=3`。这是继承的 kernel 路径问题的源码证据；本轮没有重跑旧 dlinfer 环境，不能把这里的原生→研究修复称为迁移前后重新验收。[来源核对](evidence/mtp-ray-deep-20261116/inventory/gdn-migration-provenance-r1.json)

### Ray：半秒停顿由 CPU 18 的执行机会丢失触发

CPU 探针继续保留 `submit → sleep 12ms → join → free` 顺序，补充输入/结果对象恢复、方法入口/出口、future callback、心跳、Ray C++ event stats、线程 `/proc/*/schedstat` 和运行状态。原始 `ray.init` 会为资源探测加载 ACL；之后显式传 `resources={'NPU':0}`，并保存进程 maps，确认 driver/worker 均无 Torch、torch_npu 或 ACL，仍复现。

注意：探针中的 `result_transport` 字段包含 future 在固定 sleep 后注册之前的时间，不能直接当作纯网络时延；单 rank 的 `future_to_join` 还可能在等另一 rank。最终归因使用下面的独立调度和 C++ 证据。

`ray-sched-all-r0` 的半秒事件有一致特征：关键线程处于 **R（可运行）**、采样 CPU 为 **18**，却积累数百毫秒 runqueue wait。涉及不同层次，因此外部可表现为投递慢或确认慢：

| item | 被延迟的线程 | join 等待 | runqueue 增量 |
| ---: | --- | ---: | ---: |
| 479 | driver `worker.io` | 542.033 ms | 543.177 ms |
| 1322 | driver gRPC `nexting_thread` | 522.367 ms | 532.658 ms |
| 1826 | actor 执行线程 | 512.737 ms | 521.472 ms |
| 1984 | actor RPC 线程 | 341.133 ms | 351.582 ms |
| 2330 | actor Python 事件循环线程 | 534.651 ms | 551.290 ms |

这些样本中，慢区间的 R 状态采样均落在 CPU 18；线程恢复后可能迁移到其他 CPU。采样窗口的边界会包含相邻事件，不能把所有计数增量机械分摊给一个 RPC。[关键样本](evidence/mtp-ray-deep-20261116/results/ray-cpu18-critical-samples-r1.json)、[完整调度窗口摘要](evidence/mtp-ray-deep-20261116/results/ray-sched-all-r0-sched-summary.json)

将复现进一步缩成 C++：两个 pipe 子进程、8 KiB 输入、12 ms `nanosleep`、再读取确认；**无 Python、Ray、模型或设备调用**。固定同一个 CPU 18 后，仅定时唤醒就周期性延迟约半秒：

| C++ 对照 | 迭代数 | 12 ms sleep 实际超过 50 ms | 最大实际 sleep | 停顿起点间隔中位数 |
| --- | ---: | ---: | ---: | ---: |
| CPU 18，r0 | 2500 | 15 | 565.844 ms | 2.5614 s |
| CPU 18，r1 | 2500 | 14 | 573.717 ms | 2.5605 s |
| CPU 18，r2 | 1250 | 8 | 571.409 ms | 2.5579 s |
| CPU 42，同程序/迭代数 | 2500 | 0 | 21.413 ms | — |
| 不绑定，子进程 nice=15 | 6000 | 0 | 14.461 ms | — |

CPU 42 上的独立监控器同时读取 `/proc/stat` 和 `/proc/softirqs`。例如 CPU 18 的 559.312 ms 唤醒延迟窗口对应 **55 个 softirq CPU tick（本机 CLK_TCK=100，约 550 ms）**，TASKLET 计数增加 1；其他长窗口也出现约 52–57 个 softirq tick。[对应数据](evidence/mtp-ray-deep-20261116/results/cpu18-stall-correlation-r1.json)

由此可以确认这类半秒停顿的直接原因是 **本机 CPU 18 的周期性 softirq 执行使可运行用户线程长时间得不到 CPU**，Ray 将其暴露在不同提交/回调队列上。Ray、Python、Torch 和模型都不是复现这一主机停顿的必要条件。进程不加载 ACL 不代表宿主机 Ascend 驱动被排除。

以下是边界，尚不能越过：

- TASKLET 时间相关性和 CPU 18 的 IRQ 亲和性已保存。该 CPU 同时承接 NVMe、SAS、`trans msg_chan_rx_msg_notify` 等 IRQ；仅凭名字和计数不能把长回调确定归属某个驱动。[IRQ 清单](evidence/mtp-ray-deep-20261116/inventory/irq-cpu18-r0.json)
- 容器内 `/proc/interrupts` 为空，读取目标内核栈被拒绝，tracefs 不可用。尚未获取具体 softirq/tasklet 回调栈或修复宿主机驱动。下一步需要宿主机侧的 `softirq_entry/exit`、tasklet 函数和 IRQ 时序采集。
- 没有 H200/x86 实测；不能归因为 ARM 指令集，也不能推断 H200 一定免疫。

### 本机缓解验证与剩余小停顿

仅给命令及其子进程设置 CPU 亲和性，避开 CPU 18；没有删除 await、修改同步顺序、修改 Ray 或全局 IRQ/内核设置。

| CPU-only 运行 | join 数 | >50 ms | >300 ms | 最大 join |
| --- | ---: | ---: | ---: | ---: |
| 默认探针 r0 | 6000 | 16 | 14 | 574.971 ms |
| 绑定 CPU 40–79 | 6000 | 0 | 0 | 46.783 ms |
| 仅排除 CPU 18，r0 | 6000 | 2 | 0 | 138.592 ms |
| 仅排除 CPU 18，r1 | 6000 | 3 | 0 | 120.755 ms |
| 排除 CPU 18，增加 GC 标记 | 6000 | 2 | 0 | 152.603 ms |
| 排除 CPU 18，仅诊断性关闭 driver GC | 6000 | 0 | 0 | 25.848 ms |

剩余 GC 标记运行的两次 join 分别为 152.603 / 107.111 ms，与 driver 的 generation-2 GC **150.707 / 104.523 ms** 重合；诊断性关闭 GC 后 6000 次没有 >50 ms。该探针在 driver 内存积累大量逐步记录，这部分停顿属于探针的对象/GC 开销控制，**不把关闭 GC 作为 LMDeploy 生产修复**。[GC 时间线](evidence/mtp-ray-deep-20261116/results/ray-exclude18-gc-r0-correlation.json)

单独改 gRPC poller、关闭 Ray event stats、把 worker nice 改 0、去掉 jemalloc、减少 RPC 线程池，都仍捕获 >300 ms；这些是排除或缩小假说的对照，不是相互公平的吞吐基准。strace/GDB 运行也不计性能验收；GDB 冻结导致的约 5 秒暂停有单独标记。

沿用首轮真实 HTTP 配置，只排除 CPU 18，在未改动的原生 PR 源码运行 Agents-A1 非 MTP、TP2、C1/C4 各 6 轮 + 预热、每条 256 token：**30 条测量请求及 5 条预热均完成，3618 次 RPC join 无 >50 ms，稳定测量阶段无 >100 ms SSE 间隔**。这是本机有限窗口内有效的缓解证据，不是所有尾延迟或宿主机问题永久消失的证明。[HTTP 汇总](evidence/mtp-ray-deep-20261116/results/http-exclude18-r0-summary.json)

### 复现入口、失败和未覆盖范围

本机根目录仍为 `/root/workspace/a3-mtp-ray-investigation-20261116`。发布证据提供[可执行脚本包](evidence/mtp-ray-deep-20261116/scripts.tar.gz)，各脚本亦有可读副本；`logs/*.json` 记录实际 checkout、HEAD、工作树 patch SHA、命令、环境与退出码，`run_deep.py` 另外保存每次脚本快照及继承的 CPU mask。大文件清单只列本机未上传的 tensor/syscall 原始数据及校验值，不冒充这些二进制已在 GitHub 可下载。

```bash
# 所有输出使用新 label / 文件名。以下命令在本机已有目录和隔离 venv 下运行。
cd /root/workspace/a3-mtp-ray-investigation-20261116
python scripts/run_deep.py --label gdn-recheck-new --devices 0 \
  --repo /root/workspace/a3-mtp-ray-investigation-20261116/lmdeploy-research -- \
  /root/workspace/a3-mtp-ray-investigation-20261116/scripts/gdn_state_repro_v2.py \
  --graph --iterations 200 --output /root/workspace/a3-mtp-ray-investigation-20261116/results/gdn-recheck-new.json

# 最小主机复现；同时检查实际 sleep，而不只检查消息返回时间。
g++ -O2 -std=c++17 scripts/pipe_control.cc -o /tmp/a3-pipe-control
taskset -c 18 /tmp/a3-pipe-control /tmp/a3-pipe-cpu18-new.json 2500 0
taskset -c 42 /tmp/a3-pipe-control /tmp/a3-pipe-cpu42-new.json 2500 0

# 仅改变当前命令和子进程的亲和性。CPU 18 是这台主机的观测，不应照搬其他主机。
python scripts/launch_affinity.py --exclude 18 -- python scripts/run_deep.py \
  --label ray-recheck-new -- \
  /root/workspace/a3-mtp-ray-investigation-20261116/scripts/ray_deep_probe_v2.py \
  --no-accelerators --iterations 1500 \
  --output /root/workspace/a3-mtp-ray-investigation-20261116/results/ray-recheck-new.json
```

失败/漏检均保留：首次 seed 入口漏显式导入 torch_npu；首次全量快照在 0.95 cache 预算下 OOM；首次真实输入微复现误用超出可见逻辑设备范围的 ID 8，改为逻辑 ID 2；首次 pytest 缺 `--run-ascend-hardware`；旧核一次组合 pytest 未复现，不将其改写成旧核正确，后续模块来源审计也不支持凭空归因为 checkout 选择错误，最终以固定夹具、显式锁定来源的前后回归为准；环境检查中普通 `find_spec` 被 import guard 拒绝，随后用与 guard 启动检查相同的 PathFinder 只检查包是否存在。诊断入口失败不混入迁移或模型回归。

未验证：其他模型、长上下文/高并发 MTP、MTP1 的完整模型链路、eager/prefix 关闭的完整模型矩阵、CUDA/H200、其他 CANN/Triton 版本、修复后的整模型吞吐。内核回调/驱动层根因及永久修复仍需要宿主机取证；编译器内部具体依赖错误也仍需低层 IR 分析。本轮的通过范围是上面列出的硬件状态回归、固定状态重放、Sage 21 响应重复性和本机 HTTP CPU 亲和性缓解。
