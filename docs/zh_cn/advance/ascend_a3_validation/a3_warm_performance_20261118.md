# A3充分预热性能复核（2026-11-18）

当前结论只取自本页的新配对测试；尚未完成、工作量不一致和区间过宽的项都不计为通过。
保留默认CPU亲和性（包括CPU18）、原Ray await顺序、GC和全部慢样本。旧测量叙述另行归档，原始证据不改。

## 最新状态

| 比较 | 固定源码 | 依赖 | 状态 |
| --- | --- | --- | --- |
| Agents 历史迁移 HTTP，128 token | `fb5de87a+3b909634 → 6421cefe` | `legacy` | 不确定 10项 |
| Agents 当前 PR HTTP，128 token | `fb5de87a+3b909634 → 2ccf38a4` | `stack91` | 待运行；不计通过 |
| Sage 当前 PR HTTP MTP3，128 token | `fb5de87a+3b909634 → 2ccf38a4` | `stack91` | 待运行；不计通过 |
| Agents norm pipeline，64 token | `342db85e → 5f1b1b3c` | `legacy` | 待运行；不计通过 |
| Agents norm pipeline，256 token | `342db85e → 5f1b1b3c` | `legacy` | 待运行；不计通过 |
| Sage norm pipeline，64 token | `342db85e → 5f1b1b3c` | `legacy` | 待运行；不计通过 |

## 比较与判据

- 历史Agents直接复用旧依赖，单独检验原dlinfer→native HTTP信号；当前交付两侧共同采用固定CANN9.1安装栈。
- 不跨不同profile拼接绝对吞吐来计算收益；本页每个百分比仅来自该行固定profile内的新旧配对。
- `legacy`：Triton-Ascend3.2.2 / NPUIR1.2；`stack91`：Triton-Ascend3.2.2.post2 / NPUIR2.0、NumPy1.26.4、Pandas2.2.3、Arrow25。两侧均Torch2.10.0、torch_npu2.10.0.post4、Ray2.48.0、Python3.12.13、CANN9.1.0。
- Sage历史旧编译器有状态写回错误；本轮两侧共同采用正确编译器，不把错误输出的旧速度作为性能等效认证，也不将差别全归因于预热。
- HTTP：同NPU0/1、TP2/BF16、graph、prefix ON、171输入/128输出token、greedy、thinking OFF、session262144、prefill8192、cache0.95、batch16。
- norm：完整pipeline、prefix OFF、session8192、prefill256、cache0.5、batch8、thinking默认、分别64/256输出token；不能用此替代HTTP对照。
- 每个新进程至少5个C1/2/3/4预热sweep，最后连续2个无新forward形状、graph capture、Triton compile调用或autotune bench；未直接观测厂商内部全部缓存。
- 每侧每block测25次C1和8次C4，另有5个固定资格响应。输入/输出token、cache miss、跨进程自重复和无dlinfer的原生环境均需审计通过。
- 以独立进程配对block为统计单位，10000次配对bootstrap、seed20261118；95%区间全部在±5%内才称等效，完全有利可判改善。不是“没有显著回退就算通过”。
- HTTP流式速度保留历史定义`(输出token数−1)/(末次−首次非空SSE时间)`；norm沿原pipeline定义扣除首chunk token数，不能跨两层直接混合绝对值。
- 全部失败、慢样本、启动及预热时长保留；pilot不计正式验收。

[预先判据](evidence/a3-warm-perf-20261118/ACCEPTANCE.md)与原始运行脚本、环境、命令及日志随证据归档。

## Agents 历史迁移 HTTP，128 token

Label：`agents-historical-r0`；8个全新进程配对，AB/BA各4。
工作量审计：通过。

| 并发 | 指标 | 配对变化中位数 | 配对95%区间 | 两侧block中位数之比变化 | 结论 |
| --- | --- | ---: | --- | ---: | --- |
| C1 | 总吞吐 ↑ | -2.14% | [-5.95%, -0.09%] | -3.65% | 不确定 |
| C1 | 流式速度 p50 ↑ | -3.56% | [-6.04%, +2.16%] | -4.68% | 不确定 |
| C1 | TTFT p50 ↓ | -1.64% | [-5.20%, +8.30%] | +1.32% | 不确定 |
| C1 | TTFT p95 ↓ | -1.48% | [-4.46%, +11.03%] | +3.32% | 不确定 |
| C1 | TPOT p50 ↓ | +3.70% | [-2.12%, +6.43%] | +4.92% | 不确定 |
| C4 | 总吞吐 ↑ | -2.98% | [-7.67%, +5.10%] | -2.87% | 不确定 |
| C4 | 流式速度 p50 ↑ | -1.61% | [-11.24%, +6.35%] | -4.01% | 不确定 |
| C4 | TTFT p50 ↓ | -1.52% | [-5.06%, +5.82%] | +0.31% | 不确定 |
| C4 | TTFT p95 ↓ | +3.50% | [-3.98%, +7.68%] | +1.07% | 不确定 |
| C4 | TPOT p50 ↓ | +2.17% | [-5.74%, +12.24%] | +4.26% | 不确定 |

C1总体TTFT p95：740.84 → 723.25 ms；保持配对block的总体分位数变化95%区间 [-5.20%, +3.69%]。该项与上表的逐block p95同时参与验收。


C4总体TTFT p95：2593.70 → 2613.10 ms；保持配对block的总体分位数变化95%区间 [-1.52%, +4.22%]。该项与上表的逐block p95同时参与验收。

[逐对值及统计输入哈希](evidence/a3-warm-perf-20261118/agents-historical-r0-statistics.json)；[工作量、缓存、预热和自重复审计](evidence/a3-warm-perf-20261118/agents-historical-r0-work-audit.json)。

## 本轮诊断事实与后续

历史Agents16个服务、496条请求配对全部同输入/输出token，跨进程重放也一致；每个服务完成5轮C1/2/3/4预热，测量期间没有观测到新增编译、图捕获或forward形状。原生worker没有导入dlinfer。全部请求cache miss。

当前10个主指标均未达到预先约定的等效判据，也没有确认超过5%的持续回退。旧基线自身C1吞吐的进程中位数范围为50.92–59.47token/s，说明跨进程波动仍然存在；不能据此直接把所有差异归因CPU18。

同批C4数据中，148条配对请求的流式速度下降，其中28条的首包和末包其实都更早：该指标的计时窗口会随首包时刻移动。两侧首个driver输出chunk均为1token；两侧各64批C4中，47批按请求索引0/1/2/3到达首包，另外17批有不同顺序。后64token诊断中，C4速度配对中位变化+0.12%，95%区间[−7.34%,+1.96%]，同样不足以证明±5%等效。这些诊断没有替换主指标，也没有删除慢样本。[同批窗口诊断](evidence/a3-warm-perf-20261118/agents-historical-r0-window-diagnostic.json)

norm入口pilot已通过，170输入/256输出token，engine_config与历史记录逐字段一致；不计正式性能验收。Sage新版编译器的两侧pilot、当前PR HTTP矩阵及norm三组正式矩阵继续运行。Sage旧基线pilot退出时出现MP清理超时告警，worker和driver记录齐全，相关进程随后全部退出；告警原样保留，不计成推理期间失败，也不宣称生命周期无问题。

[完整历史HTTP请求/输出与脚本快照归档](evidence/a3-warm-perf-20261118/agents-historical-r0-raw.tar.gz)含全部正式测量和预热样本、命令元数据及每次运行的工具快照。[归档清单](evidence/a3-warm-perf-20261118/archive-manifest-r0.json)给出成员/归档哈希及完整服务日志的本机路径和哈希；完整服务日志留在A3，不在此包中。

本轮尚无可替换旧结论的性能通过项，旧结果暂仍保留；通过项将按用户要求更新正文，并将对应旧性能叙述移到old/。未验收项继续明确显示。
