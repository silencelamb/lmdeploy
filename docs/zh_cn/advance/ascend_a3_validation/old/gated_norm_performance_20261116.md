# 历史性能记录（已归档）

以[最新A3性能结果](../a3_warm_performance_20261118.md)为当前结论；以下保留原测量事实，不代表当前验收状态。相对引用已调整，原始数据未改。

# Agents-A1 gated RMSNorm 性能复查

日期：2026-11-16，Asia/Shanghai。原始日志与命令元数据保留 UTC。

## 结论与比较范围

此前 Agents-A1 的 C4 吞吐 −5.15%、C1 TTFT +14.15% **没有在同进程、交替顺序的对照中稳定复现**。
实际使用的新旧 gated norm NPU 二进制逐字节相同；目前证据不支持把这两个百分比认定为
`RMSNormGated` 重构导致的算子性能回归。与此同时，流式生成确实存在数百毫秒的间歇停顿，
不能把“未确认 norm 回归”扩大成“全部性能问题已解决”。

本轮比较的是两个**已经原生迁移**的版本：

| 对象              | 固定版本                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| gated norm 重构前 | [342db85e](https://github.com/silencelamb/lmdeploy/commit/342db85ecff3c0dc864a4bf8613c589c270f4141) |
| 当前 PR           | [5f1b1b3c](https://github.com/silencelamb/lmdeploy/commit/5f1b1b3c12b1c198813f65275f778c7ad292fb02) |
| 开发分支对应实现  | [7d984d0a](https://github.com/silencelamb/lmdeploy/commit/7d984d0abce61849237d436ac4243fc964e93e7b) |

旧 LMDeploy + dlinfer 的迁移比较仍以[历史报告](../migration_validation.md)为准，不将本轮两侧共有的
现象追溯成旧 dlinfer 基线也已验证。精度、模型覆盖和首次重构测试见[重构报告](../gated_norm_refactor_20261115.md)。

本轮没有修改 PR、开发分支、模型权重、依赖环境或生产 GC 策略。所有计时与替换实验均位于验证目录，
父进程和 worker 继续使用禁止发现/导入 dlinfer 的 guard。

## 指标与预先约定

- Agents-A1、BF16、TP2、A3 0/1 两张卡、NPUGraph decode、非 MTP、无 prefix cache。
- `max_prefill_token_num=256`、`session_len=8192`、`max_batch_size=8`、KV cache fraction 0.5。
- C1/C4 分别表示一次提交 1/4 个请求，均使用两张卡；每条输入 170 token，输出固定 256 token。
- throughput 是整个批次的输出 token 总数除以从提交到全部完成的时间，包含 prefill、调度和结果传递。
- TTFT 是客户端收到首 token 的时间；先对同批请求取中位数，再对各次测量取中位数。
- 每请求 decode rate 从首个到末个输出片段计算，也包含中途调度等待。不能当作纯 NPU kernel 速度。
- 吞吐下降超过 5% 或 TTFT 增长超过 10% 触发确认；不是单轮定罪/认证阈值。

[验收与追加诊断约定](../evidence/norm-perf-20261116/ACCEPTANCE.md)在对应实验之前保存。
这是 Python pipeline 流式测试。

## 逐层定位证据

### 1. 实际编译的 kernel 完全相同

从本轮独立 Triton cache 中找到当前实现与冻结旧实现各一个实际使用的 specialization。
两份 `.npubin` 均为 4928 字节，直接 bytes 比较相等，SHA256 同为：

```text
e6fe0b4492b0461af95bc9746c24c218b2fb5a8d447018042d961b731114d605
```

该结论覆盖 Agents 实际使用的 BF16、width 128、SiLU、affine、无 bias、norm-before-gate 路径；
行数是运行时参数。它不扩展到 sigmoid、其他 dtype 或其他 specialization。
两个 cache 的 IR 分别指向旧/新源码，compiler metadata、IR、大小、哈希和比较脚本均保存于
[binary audit](../evidence/norm-perf-20261116/results/kernel-binary-audit.json)及其关联证据。

### 2. 同一模型、同一对 worker 中交替新旧实现

一次加载权重，30 层 norm 在旧 Module/kernel 与当前公共层间切换；每对模块直接共享同一个已加载
Parameter，各自保留独立的 graph cache 和 graph pool。两种模式、C1/C4 均先预热，再做每场景
8 对 AB/BA 交替测量，成对请求完全相同。原始样本全部保留，不剔除慢样本。

正式的 GC 观察对照结果如下。百分比为两侧中位数之比；另给出成对比值的中位数和 bootstrap
区间，以显示小样本波动，不能把偶然的正收益当作重构加速。

| 并发 | 指标                   |  旧实现 |  新实现 | 中位数变化 | 成对变化中位数 | 成对中位数 95% bootstrap 区间 |
| ---- | ---------------------- | ------: | ------: | ---------: | -------------: | ----------------------------- |
| C1   | throughput，token/s    |  70.480 |  69.491 |     −1.40% |         +4.10% | \[−11.35%, +14.90%\]          |
| C1   | TTFT，ms               | 242.334 | 229.994 |     −5.09% |         −4.17% | \[−9.49%, +4.93%\]            |
| C1   | 每请求 decode，token/s |  75.092 |  73.721 |     −1.83% |         +4.06% | \[−13.63%, +17.28%\]          |
| C4   | throughput，token/s    | 201.642 | 223.580 |    +10.88% |         +4.68% | \[−1.09%, +15.59%\]           |
| C4   | TTFT，ms               | 594.566 | 560.297 |     −5.76% |         −4.31% | \[−14.32%, +5.41%\]           |
| C4   | 每请求 decode，token/s |  57.406 |  63.361 |    +10.38% |         +5.21% | \[−1.27%, +15.69%\]           |

此前第一次切换实验的预热后 C4 吞吐中位数变化为 +0.53%，成对中位数为 −0.18%，也没有稳定复现
原来的 −5.15%。但该次实验最终因首轮 capture 输出断言失败而退出，**不把它计为通过**；只保留为
诊断记录。正式实验的第二轮预热、全部测量和诊断阶段共 55 对响应、14080 个 token ID 全部相同，
流式性能测量没有采集 logprob。

数据：[完整成对记录](../evidence/norm-perf-20261116/results/agents-paired-gc-observe.json)、
[统计与置信区间](../evidence/norm-perf-20261116/results/agents-paired-gc-observe.summary.json)。

### 3. 正常 decode 的 graph 路径和设备耗时没有整体拉长

性能测量结束后，单独采集 NPU event 与主机提交时间。每个模式、每个并发、每个 rank 均有
512 次 decode forward，全部走已有 graph 重放，诊断期没有新 capture。

| rank 0 的模型 forward 设备区间中位数 |     旧实现 |     新实现 |
| ------------------------------------ | ---------: | ---------: |
| C1 decode                            |   9.986 ms |  10.102 ms |
| C4 decode                            |  12.716 ms |  12.668 ms |
| C1 prefill                           | 214.400 ms | 216.700 ms |

这是整个模型 forward 的设备事件区间，不是 gated norm 单算子时间。它可以包含队列或跨卡等待。
一次新实现 C4 的 rank 0 区间为 551 ms，同时 rank 1 在进入相应 forward 前停了约 551 ms；
不能把这个离群点解释成 norm kernel 执行了 551 ms。

C4 每轮实际有四次 170-token prefill。256-token 的 prefill 上限容不下两条完整请求，所以早完成
prefill 的请求需要等其余请求进入 decode；原记录中首 token 后约 0.6–0.8 秒的间隔包含这段工作。
在比较纯 decode 时必须将它与稳定 decode 中途的停顿分开理解，原始吞吐统计仍保留完整时间。

### 4. 长停顿与 GC 不重合，已出现明确的主机提交等待

在主进程和两个 worker 安装 GC callback，记录 generation、起止时间、耗时与 PID。
该次实验共发现 28 个首 token 之后、超过 80 ms 的 row-0 decode 间隔，均没有与所记录的 GC
区间重合；正常 GC 也未出现能解释这些数百毫秒间隔的耗时。因此本轮没有依据将停顿归因于 Python GC。

随后保留原模型与 norm，只给输入提交、metadata、模型 forward、采样、输出复制加分阶段计时。
在 C4 第 0 次测量、第 149 个 token 后复现 582.308 ms 的间隔，其中
`RayExecutor.forward_async` 提交输入阶段耗时 **562.143 ms**，该线程实际 CPU 时间仅 **3.445 ms**，
Linux runqueue 等待增量为 0 ms。对应时间不是 CPU 在执行 norm，也没有落在模型计算阶段。

相关代码是 [RayExecutor.forward_async](https://github.com/silencelamb/lmdeploy/blob/5f1b1b3c12b1c198813f65275f778c7ad292fb02/lmdeploy/pytorch/engine/executor/ray_executor.py#L582)。
它与 Ascend graph runner 在本轮新旧 commit 之间均无 diff。分阶段结果见
[stage profile](../evidence/norm-perf-20261116/results/agents-stage-profile.json)。

追加一轮计时，仅对 `forward_async()` 中创建的 `asyncio.gather(*self._prev_out)` Future
增加完成回调，并同时监测 `ray.put`、`ray.internal.free` 与 actor 提交调用。复现结果如下：

| 场景                | token 间隔 | 等上一条 worker 提交 RPC 返回 | 该线程 CPU 时间 | runqueue 等待增量 |
| ------------------- | ---------: | ----------------------------: | --------------: | ----------------: |
| C1 r1，token 165 后 | 550.898 ms |                    533.124 ms |        0.932 ms |              0 ms |
| C1 r2，token 184 后 | 514.060 ms |                    495.721 ms |        0.749 ms |              0 ms |
| C4 r3，token 23 后  | 346.535 ms |                    340.259 ms |        0.738 ms |              0 ms |
| C4 r5，token 6 后   | 244.939 ms |                    221.728 ms |        1.090 ms |              0 ms |

这四次停顿主要时间都落在[等待上一条 Ray 提交任务的返回](https://github.com/silencelamb/lmdeploy/blob/5f1b1b3c12b1c198813f65275f778c7ad292fb02/lmdeploy/pytorch/engine/executor/ray_executor.py#L592)。
这里的 `worker.forward_async` 是把输入交给后台模型循环的 RPC，不等于整次模型计算本身。
同期 `ray.put/free`、actor 提交没有超过 50 ms 的记录，且两侧共享此源码。
[逐次重合关系与源码 blob 核对](../evidence/norm-perf-20261116/results/ray-wait-localization.json)保存了
绝对单调时间戳和两个 commit 的相同 blob ID；[追加原始记录](../evidence/norm-perf-20261116/results/agents-await-profile.json)
保存全部阶段数据。

**定位结论是：已复现的长停顿发生在主机等待前一条 Ray RPC 返回的阶段。** Ray 内部为什么延迟
通知/调度尚未锁定，没有声称修复，也没有通过移除等待来改变执行顺序。进一步调查应记录各 rank
RPC 的入队、开始、返回和 driver 唤醒时间，区分 worker 调度与结果传输。
上述证据只定位本轮观察，不能凭它替代旧 dlinfer 环境的性能诊断，也没有证明原来五次测量的
每一次停顿都是同一个触发源。

## 诊断限制与原始失败

1. 同进程动态替换 norm 并为两种模式分别首次 capture 时，首轮 capture 请求从 token index 1 起
   未对齐。第一次实验的最终断言因此失败。后续实验单列 `capture_warmup` 并完整记录两次失败比较，
   第二次预热及后续全部成对输出相同。首次 capture 差异没有独立归因，不计入精度通过；这个改装
   诊断实验只支持预热后性能定位，原源树的精度结论仍以已有完整 trace 为准。
2. 第一次运行使用了错误的相对脚本路径；第二次在保存枚举类型配置时失败。两个启动失败日志均保留，
   第三次才完成首轮计时。没有将失败记录覆盖成成功记录。
3. Python py-spy 栈采样保留原始数据及采样错误数，只作线索；后一次 `--native` 在这台平台报告
   unsupported，未获得 native 栈，三个失败日志也保留。它们不充当成功的 C++ profiling 证据。
4. 只有两个 rank、一个模型、固定并发和较短重复；完整生产 soak、不同负载和六模型 GPU 对照仍未完成。
   置信区间较宽，没有宣称性能全面通过或已修复全部停顿。

## PR 与 CI 记录

[PR #5134](https://github.com/InternLM/lmdeploy/pull/5134) 标题改为
`feature(ascend): migrate A3 support into the native PyTorch backend`，描述中的比较 commit 和相关 PR
均改为链接。本报告和证据只进入 fork 文档分支。

本次顺便复核现有 [A100 unit-test job](https://github.com/InternLM/lmdeploy/actions/runs/43727003277/job/132851013103)：
新增 `test_rms_norm_gated.py` 的 83 项全部通过，包含 56 项真实 CUDA/graph 检查；模型构造/加载文件
8 项通过。整个 job 为 6424 passed、3 failed、400 skipped，三项失败为 `BaseModelAgent` fixture
缺 `cache_config`。ETE workflow 成功；CI 整体仍非全绿。A100 算子检查不能替代六个完整 GPU 模型的
按位比较，后者仍待用户在 GPU 平台执行。

## 复现与证据

本地目录：`/root/workspace/a3-norm-perf-20261116`。
[证据清单及 SHA256](../evidence/norm-perf-20261116/index.json)包含命令、环境差异、源码状态、
完整样本、阶段计时、GC 时间记录、源码/编译产物核对及 Markdown 形式的脚本快照。
大依赖环境、工具二进制和整个 Triton cache 留在本机；原始实验数据均按成功/失败实际状态保存。

使用新的 label 和输出名，避免覆盖历史数据：

```bash
python /root/workspace/a3-norm-perf-20261116/scripts/run.py \
  --label agents-paired-gc-repeat --devices 0,1 --timeout 1200 -- \
  /root/workspace/a3-norm-perf-20261116/scripts/paired_norm_gc.py \
  --gc-policy observe \
  --output /root/workspace/a3-norm-perf-20261116/results/agents-paired-gc-repeat.json

python /root/workspace/a3-norm-perf-20261116/scripts/analyze_pairs.py \
  /root/workspace/a3-norm-perf-20261116/results/agents-paired-gc-repeat.json
```

`run.py` 固定 native 环境、工作目录、guard、设备与源码记录；`perf_worker_gc.py` 从冻结旧工作树
加载旧 Module/kernel，不导入 dlinfer。首轮 capture 的限制必须按上文保留，不能只摘取成功的后续输出。
