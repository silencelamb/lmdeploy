# MTP 贪婪验证与在线 Ray RPC 停顿定位

日期：2026-11-16，Asia/Shanghai；原始命令、日志时间保留 UTC。

本轮回答两个问题：MTP 输出不一致是否误用了随机拒绝采样，以及此前 pipeline 中的 Ray 等待是否也会影响在线请求。
**MTP 本轮实际执行了贪婪验证，分歧前 target logits 已经不同；真实 HTTP 路径也捕获到一次约 484 ms 的同步停顿，主要时间发生在 worker 进入输入提交方法之前。两个问题的最终触发原因尚未解决。**

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
