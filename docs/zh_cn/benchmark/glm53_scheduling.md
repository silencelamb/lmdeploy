# GLM-5.3-Flash 调度优化的基线

本页记录改动前可以区分执行和调度影响的测量。单节点 8×H200，DP8 + EP8 + TP1。LMDeploy 的线上组合使用 DeepGEMM MegaMoE 和可选 FlashKDA。两组的准入、GPU 等待和前端流式输出均计入服务时钟。[机器可读结果](../../../benchmark/glm53_mega_moe/results/h200_scheduling.json)

## MTP0 的生成窗口与请求补入

8K 输入、256 输出、C=64。把首批 64 个请求按开始时间排序，确认每 rank 有 8 条，各自已产生首 token，且整个测量区间内没有请求完成或新请求补入。取每条请求第 33～224 个输出 token 的 191 个间隔。两轮 LMDeploy 平均 16.26 ms/token，同机 vLLM `5328ae2fef` 为 14.53 ms/token。收窄到第 65～192 个 token 可以复核结论。脚本在这些边界不成立时直接报错。[窗口提取代码](../../../benchmark/glm53_mega_moe/decode_window.py)

| 8K/256、C=64，滚动 256 个请求 | LMDeploy | vLLM |
| --- | ---: | ---: |
| 总输出 tok/s | 1,702 | 1,948 |
| 平均 TTFT | 3.287 s | 1.481 s |
| 首次调度前排队 | 2.084 s | 0.626 s |
| 首次调度到首 token | 1.192 s | 0.835 s |
| 首 token 到完成 | 6.313 s | 6.890 s |

首次调度前的差值占平均 TTFT 差距的 80.8%。从首 token 到完成的时长还包括其他请求进入 prefill、batch 改变及等待下一次前向的时间。因此，纯生成窗口和滚动服务必须分别比较。LMDeploy 每档两次完整成功启动，vLLM 为本轮一次。

## 带 MTP 的验收

MTP5 的另一组配对试验在相同验证深度下出现了不同的瓶颈。8K/256、C=64 的两边每轮可提交约 4.61 / 4.63 个 token。平均排队仅差 0.29 秒，首 token 到完成却差 3.34 秒。C=32 时，在至多 32 条探针都输出 32 个 token 后，再投递 24 条冷 8K prompt。在 LMDeploy 中，50 对探针连 512 个输出 ID 都完全一致，TPOT 的配对中位差仍为 +3.82 ms。此处改变的是现有服务上的负载；新调度的收益需要同引擎开关实验。

下一轮对同一 rank 的混合批按实际 query / verify token 数计入预算。MTP0、MTP3、MTP5 的验证宽度最多为 1、4、6，而接受后的实际提交量由运行时计数给出。优先验收上线候选 MTP3，指标同时覆盖 TTFT p95、TPOT p95、最长流式停顿、整机输出 tok/s 与正确的 prefix/compact-state 恢复。
