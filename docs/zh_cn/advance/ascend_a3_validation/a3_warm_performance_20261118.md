# A3充分预热性能复核（2026-11-18）

当前结论只取自本页的新配对测试；尚未完成、工作量不一致和区间过宽的项都不计为通过。
保留默认CPU亲和性（包括CPU18）、原Ray await顺序、GC和全部慢样本。旧测量叙述另行归档，原始证据不改。

## 最新状态

| 比较 | 固定源码 | 依赖 | 状态 |
| --- | --- | --- | --- |
| Agents 历史迁移 HTTP，128 token | `fb5de87a+3b909634 → 6421cefe` | `legacy` | 不确定 10项 |
| Agents 当前 PR HTTP，128 token | `fb5de87a+3b909634 → 2ccf38a4` | `stack91` | 尚未完成：1/8配对；不计通过 |
| Sage 当前 PR HTTP MTP3，128 token | `fb5de87a+3b909634 → 2ccf38a4` | `stack91` | 3项计时区间达到等效带；draft入口覆盖待补核，暂不最终验收 |
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
- 本轮前8轮交错执行C1/C4，其后继续C1；同一配对两侧顺序一致。历史首轮按并发分组执行，因此本轮不是“只改变预热”的单因素因果实验。
- 以独立进程配对block为统计单位，10000次配对bootstrap、seed20261118；95%区间全部在±5%内才称等效，完全有利可判改善。不是“没有显著回退就算通过”。
- HTTP流式速度保留历史定义`(输出token数−1)/(末次−首次非空SSE时间)`；norm沿原pipeline定义扣除首chunk token数，不能跨两层直接混合绝对值。
- 全部失败、慢样本、启动及预热时长保留；pilot不计正式验收。
- 原MTP forward计数只覆盖target，draft另走独立入口；全局编译/graph捕获记录保留，draft形状直接观测另做同配置补核。按phase记录的driver输出槽数可能包含上一批迟到输出，不能直接解释为逐请求接受率。

[预先判据](evidence/a3-warm-perf-20261118/ACCEPTANCE.md)与原始运行脚本、环境、命令及日志随证据归档。

## Agents 历史迁移 HTTP，128 token

Label：`agents-historical-r0`；8个全新进程配对，AB/BA各4。
输入/输出及核心环境审计：通过。

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

## Sage 当前 PR HTTP MTP3，128 token

Label：`sage-current-r0`；8个全新进程配对，AB/BA各4。
输入/输出及核心环境审计：通过。
draft入口覆盖补核尚未完成；表内满足等效区间的项暂不发布最终通过。

| 并发 | 指标 | 配对变化中位数 | 配对95%区间 | 两侧block中位数之比变化 | 结论 |
| --- | --- | ---: | --- | ---: | --- |
| C1 | 总吞吐 ↑ | -2.13% | [-4.49%, +2.36%] | -1.87% | 计时等效，待draft补核 |
| C1 | 流式速度 p50 ↑ | -1.67% | [-4.74%, +1.54%] | -1.64% | 计时等效，待draft补核 |
| C1 | TTFT p50 ↓ | +0.49% | [-8.02%, +4.81%] | +0.16% | 不确定 |
| C1 | TTFT p95 ↓ | -1.53% | [-10.52%, +41.73%] | +1.82% | 不确定 |
| C1 | TPOT p50 ↓ | +1.71% | [-1.52%, +4.97%] | +1.66% | 计时等效，待draft补核 |
| C4 | 总吞吐 ↑ | -0.16% | [-7.40%, +5.47%] | -2.34% | 不确定 |
| C4 | 流式速度 p50 ↑ | -1.99% | [-10.86%, +6.33%] | -5.05% | 不确定 |
| C4 | TTFT p50 ↓ | +1.72% | [-8.52%, +5.31%] | -0.51% | 不确定 |
| C4 | TTFT p95 ↓ | -1.29% | [-5.75%, +7.02%] | +1.66% | 不确定 |
| C4 | TPOT p50 ↓ | +2.57% | [-5.86%, +11.57%] | +4.84% | 不确定 |

C1总体TTFT p95：758.50 → 751.01 ms；保持配对block的总体分位数变化95%区间 [-7.94%, +44.03%]。该项与上表的逐block p95同时参与验收。


C4总体TTFT p95：2711.40 → 2679.48 ms；保持配对block的总体分位数变化95%区间 [-9.21%, +7.47%]。该项与上表的逐block p95同时参与验收。

[逐对值及统计输入哈希](evidence/a3-warm-perf-20261118/sage-current-r0-statistics.json)；[工作量、缓存、预热和自重复审计](evidence/a3-warm-perf-20261118/sage-current-r0-work-audit.json)。

## Sage补核范围

Sage的8个配对进程、496条请求配对和全测量重放一致，固定源码fb5de87a+dlinfer3b909634对2ccf38a4，共同使用stack91。三个C1计时区间达到等效带，其余7项仍不确定。直接draft计数缺口属于观测工具问题；原始编译/graph记录及输入输出数据保留。将在当前Agents矩阵完成后，用两个全新服务重复相同预热和完整C1/C4请求序列，直接记录draft_model_forward形状并匹配原b00输出。补核的计时不并入8配对统计。

SpecAgent及base/deepseek_mtp/qwen3_5_mtp proposer四个文件在两侧SHA完全相同。首个draft forward保留target输入形状；后续两个AR forward使用每序列一个token。非末尾prefill chunk跳过后续两个AR forward。直接观测用于补齐这条已阅读的源码路径，不能把未记录的draft计数写成已测。[执行计数与源码核对](evidence/a3-warm-perf-20261118/sage-current-r0-execution-counts.json)

按阶段采集的driver槽数存在迟到输出：某些C1阶段出现上一批的多行结果。因此没有用槽数差异宣称MTP接受率下降。事后按前一批并发分层，Sage原生的>1s TTFT为C4之后2/40、C1之后6/160，不支持“只有并发切换才触发”的假设；这一分层不删除样本，也不替代主统计。[前序批次诊断](evidence/a3-warm-perf-20261118/sage-current-r0-previous-batch-diagnostic.json)

[已发布的历史Agents完整请求、命令与工具快照](evidence/a3-warm-perf-20261118/agents-historical-r0-raw.tar.gz)及[归档哈希清单](evidence/a3-warm-perf-20261118/archive-manifest-r0.json)保持不变。Sage完整原始日志和请求文件现保存在A3的`/root/workspace/a3-perf-warm-20261118`，当前先发布汇总与审计，后续一并补充归档。

尚未发布任何最终性能通过项；通过项补核完成后更新正文，并将对应旧性能叙述移至old/。未验收项继续明确显示。
