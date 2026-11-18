# PR #5134 性能验证：最新结果与覆盖范围

更新日期：2026-11-18。A3采用[充分预热后的新实测](a3_warm_performance_20261118.md)，旧性能叙述移入文末归档。H200采用11月18日新实测，保持各自固定版本和既定判据。

**不能把现有结果概括为“所有性能都对齐”。** H200已测配置（含动态HTTP C4）的独立性能均未触发持续回退；精度限制单列。这不是±5%统计等效检验。A3的当前HTTP C1已有通过项；C4、部分TTFT与norm仍未完成等效验收，最新状态见A3节。

| 比较 | 固定范围 | 可支持的结论 |
| --- | --- | --- |
| H200 迁移回归 | `442097e8` → `f64ab506`；`2ccf38a4` 的推理/测试/依赖源码树相同 | 合格且完整预热的对照未见持续明显回退；不是所有指标的等效性证明 |
| A3 历史迁移复测 | LMDeploy `fb5de87a`＋dlinfer `3b909634` → native `6421cefe`，旧依赖 | Agents8配对仍未证明±5%等效 |
| A3 当前交付 | 同一旧源码 → `2ccf38a4`，两侧匹配stack91 | Sage/Agents的指定C1指标通过；C4和TTFT仍有未验收项 |
| A3 norm 重构 | 已原生迁移的 `342db85e` → `5f1b1b3c`，具体替换实验见原报告 | 长/短生成均已复测；norm64有两项C1通过和需定位的吞吐/TTFT信号；不替代HTTP验收 |
| A3 circular GDN / 编译器控制 | 旧/新编译器、stage1/stage3 的独立控制 | 微内核和冻结 target 主干结果单列；不等于完整 MTP HTTP 性能验收 |

## H200

2026-11-18的新实测已完成。34个短工作负载/评分配置（含动态HTTP C4）、两个GLM 32K保序迁移控制及16组共享算子配置，均未触发预先约定的持续性能回退判据。精度与性能按用户要求分别报告。

| 完整模型 | 当前结论 |
| --- | --- |
| GLM-5.3-Flash | 非MTP pipeline/HTTP及MTP3 pipeline的C1/C4未见持续回退 |
| Intern-Decision-4B | 真实评分pipeline/HTTP的C1/C4通过；MTP不适用 |
| Sage-A1.5 | 非MTP pipeline/HTTP及MTP3 pipeline的C1/C4通过 |
| Agents-A1 | 非MTP pipeline/HTTP及MTP3 pipeline的C1/C4通过 |
| Intern-S2-Preview-397B-FP8 | 非MTP pipeline/HTTP及MTP3 pipeline的C1/C4未见持续回退 |
| GLM-5.3（保序控制） | 非MTP短请求pipeline/HTTP、MTP3 pipeline及32K控制未见持续回退 |

性能通过指本轮既定持续回退门槛，不是±5%统计等效证明。每格五对全新进程AB/BA、完整预热、同输入/输出长度与cache条件，全部计时外完整token/top-k检查及资源记录均保存。动态HTTP C4的Sage/Agents token/top-k及Decision评分逐位一致；Flash/Intern-S2/GLM的数值重复性限制保留，独立性能结论另列。[具体含义和三项C4数据](http_c4_performance_only_20261118.md)。

**预热要点：**先完成引擎全部解码图预热，再用实请求覆盖测量形状；HTTP还要覆盖完整C4 prefill/tuning形状。Triton核对实际引用产物哈希，不只核对目录变量。

完整新数字、逐对区间、GLM保序选项独立成本及复现资料见[最新H200性能报告](cuda_warm_performance_20261118.md)。旧百分比已移至文末归档。

## A3

当前PR `2ccf38a4` 的真实HTTP C1已通过：Sage MTP3总吞吐、流式速度p50、TPOT，以及Agents总吞吐、TPOT。两组各8个全新进程配对、496条对应响应逐token一致；Sage直接draft形状补核通过。独立norm重构对照（342db85e→5f1b1b3c、64输出token）的C1流式速度和TPOT也通过±5%等效判据。

**不能宣布所有性能持平。** norm64出现需优先定位的不利信号：C4总吞吐配对中位−5.35%，95%区间[−7.82%,−2.92%]，7/8配对下降；C1 TTFT+6.25%，区间[+2.87%,+9.33%]，7/8上升。本轮区间支持不利方向，退化幅度是否稳定超过5%以及具体代码归因仍待核验。它们不能用其他通过指标关闭。

六个主矩阵均已完成，输入输出、缓存、固定版本及重复性检查通过。HTTP C4、其他TTFT及norm仍有未验收项；独立主机/设备计时正在执行。完整逐项区间与证据以[最新A3性能结果](a3_warm_performance_20261118.md)为准。旧性能叙述已移至old/，原始数据保留。

## 证据与历史追溯

新A3实验的判据、固定版本、逐对数据与复现工具见[最新报告](a3_warm_performance_20261118.md)。H200部分保留其独立固定版本与原判据，不用A3结果代替。

旧A3表格、11月17日的CPU数据复核及其原始数字见[历史归档](old/performance_review_before_warm_20261118.md)。原始证据未删除，当前未关闭项在新主表中明确显示。

H200旧性能表见[历史归档](old/h200_performance_review_before_rerun_20261118.md)；当前结论以新报告为准。
