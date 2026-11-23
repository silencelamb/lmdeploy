# GLM-5.3-Flash 同一 rank 的混合前向

部署验收使用 GLM-5.3-Flash、8×H200、DP8 + EP8 + TP1、MTP3 和 RDMA Mooncake。目标是在 AgentX C90 降低 TTFT p95，同时保住 TPOT p50/p95 和整机输出吞吐。

## 怎样读两行对照

**已发布的对齐版**是此前已推到 `silencelamb/lmdeploy` 的调度提交 `8ff9e72d`。8 个 rank 共同参与专家并行计算。调度器沿用 worker 已有的集体前向轮次编号，尽量让忙碌 rank 在同一相位接纳 prefill，周期设为 2。一次含 prefill 的集体轮次可以同时服务更多 rank，其余轮次给在途请求做 decode 或 MTP 验证。空闲 rank 可以立即接单，忙碌 rank 也有等待上限。固定负载里，每个含 prefill 的轮次平均处理 prefill 的 rank 数从独立轮转时的 2.44 增至 3.60–3.91。[调度提交](https://github.com/silencelamb/lmdeploy/commit/8ff9e72dbe39f528c2b4da1ffa8a15ff23fde9bc) · [轮次测量](glm53_scheduling.md)

**同代码关闭混批**是在含混批实现的工作树中设置 `LMDEPLOY_GLM53_MIXED_FORWARD=0`，保留 4K 忙时预算和周期 2 的跨 rank 对齐。每个 rank 一次前向选择 prefill 或生成验证。不同 rank 在同一个集体轮次可以各做不同的工作。表中这两行分别来自独立冷启动，TTFT p95 为 3,602.19 和 3,603.04 ms。

| 数值 | 实际含义 |
| --- | --- |
| 4K | 两行对照都用 `LMDEPLOY_INTERLEAVED_PREFILL_TOKENS=4096`。某个 rank 已有在途生成时，新准入的文本 prefill 一轮最多使用 4,096 个 token，前缀命中后按剩余计算量计费。 |
| 8K | 两行都用 `--max-prefill-token-num 8192`。这是引擎每个 rank 一次前向的总预算，空闲 rank 可用满 8,192 个 token。 |
| 8K/256、C64 | 这是单独的固定负载，每条请求输入 8,192、输出 256 个 token。AgentX C90 使用 Weka 256K 轨迹，输入长度随请求变化。 |

同一对齐配置的固定回放有一个直接例子。冷首轮日志中八个 rank 各处理 8,192 个 token。到第 4 个集体轮次，三个做 prefill 的 rank 各处理 4,096 个 token。关闭同 rank 混批时，同一个 rank 仍可把多条在途请求放在同一次 MTP 验证里。

## 一个前向里做了什么

1. Host 在前缀匹配后确定本轮 prompt 行和 KV 预留，同时给已经在途的生成行预留实际验证宽度。MTP0 每行 1 个 query，MTP3 每行 4 个 query。每个混合前向的总 query 数受引擎的 8,192-token 上限约束。模型不支持的请求组合沿用已有调度。
2. Worker 从保留的 `StepInputs` 取最新 decode token、草稿、位置和状态库行号。目标模型只调用一次。mHC、线性投影、FFN 和跨卡 MoE 处理拼在一起的 token。KDA 和 KPool/MLA 使用各组自己的 prefill 或 decode 元数据访问缓存。输出层在同一次投影中计算每个 prompt 最后一个位置及在途行的全部验证位置。
3. MTP3 为新 prompt 和在途请求分别移位。非末尾 prompt chunk 使用已知的下一个输入 token，验证行根据实际拒绝数写入替换或奖励 token。两组共用草稿模型的批次。Host 按独立完成编号确认两组 compact-state 和 checkpoint，先推进原有行，再把完成 prefill 的新行加入下一轮。采样、停止和交付均保留请求自己的行映射。

这一实现面向 CUDA 上的 GLM-5.3 混部文本生成、attention TP1、MTP0 或三草稿 DeepSeek MTP。纯 decode 继续使用原有 CUDA Graph。输入打分、返回全部 logits、特殊适配器、受约束生成与多模态输入等组合选用原先的前向路径。[调度和模型实现](https://github.com/silencelamb/lmdeploy/commit/435130da)

## AgentX C90 实测

| C90，MTP3 + Mooncake | TTFT p50 / p95 ms | TPOT p50 / p95 ms | 输出 tok/s | Goodput req/s | 达标 / 正式可评分请求 | 正式空内容请求 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 已发布的对齐版 | 726.71 / 3,602.19 | 22.45 / 58.13 | 1,433.65 | 0.9277 | 1,707 / 3,042 | 0 |
| 同代码关闭混批 | 733.94 / 3,603.04 | 21.75 / 53.21 | 1,479.86 | 0.9587 | 1,764 / 3,069 | 0 |
| 4K 混批 | 780.47 / 3,683.09 | 23.32 / 56.72 | 1,391.44 | 0.8603 | 1,583 / 2,965 | 2 |
| 长后缀扩块混批 | 796.16 / 3,628.92 | 23.78 / 67.29 | 1,377.98 | 0.7967 | 1,466 / 2,929 | 0 |
| 只在长后缀借验证轮混批 | 786.74 / 3,442.63 | 21.83 / 65.90 | 1,414.47 | 0.9342 | 1,719 / 2,998 | 2 |

各窗口采用相同的 QA 服务参数、Weka 256K、seed 20260707、RDMA Mooncake 160 GiB/worker，分别独立冷启动和正式计时 1,800 秒。表中 TTFT、TPOT、输出吞吐和 goodput 都取 AIPerf 有效导出。普通 4K 混批与选择性混批各有两条正式请求被客户端判为空内容，单独列在最后一栏。

**C90 保留 4K 跨 rank 对齐、关闭同 rank 混批作为上线参考。** 相对同代码关闭混批的复测，选择性候选将 TTFT p95 从 3,603.04 降至 3,442.63 ms，降低 160.41 ms（4.45%）。TPOT p50 从 21.75 变为 21.83 ms，TPOT p95 从 53.21 升至 65.90 ms（+23.85%），输出吞吐从 1,479.86 降至 1,414.47 tok/s（−4.42%）。普通 4K 和长后缀扩块的 TTFT p95 均高于对齐参考。三种混批候选均未同时达到更低 TTFT p95、保住生成时延及吞吐的目标。[各窗口原始汇总与配对结果](../../../benchmark/glm53_mega_moe/results/h200_mixed_forward.json)

选择性候选正式窗中有 2,998 条可评分请求，1,719 条逐请求达标。正式请求覆盖的时间段内，8 个 rank 均记录到混合目标前向。两边共同完成评分的 2,975 个 turn 上，TTFT p95 为 3,701.81 / 3,449.79 ms，TPOT p95 为 53.34 / 65.71 ms。按 107 条源轨迹整组重采样 1,000 次，候选减去关闭对照的 TTFT p95 差值区间为 −521 至 +179 ms，TPOT p95 差值区间为 +7.37 至 +17.72 ms。此 95% 区间描述这批共同轨迹的抽样变化。每个候选只有一个独立正式窗。

普通 4K 窗口两条空内容记录对应的服务端请求均以长度上限结束，分别生成 1,094 和 291 个 token。对齐复测和本次选择性候选都在计分结束后，各重放了这两条原始数据集 turn 两遍。每组四次都交付了可解析内容并达到指定输出长度。重放使用独立请求上下文，原始失败流的具体原因仍未确定。选择性正式窗的空内容计数仍按本次导出保留。

在已发布对齐版与 4K 混批的 2,956 个共同评分 turn 上，TTFT p95 从 3,640.29 变为 3,691.29 ms，TPOT p95 从 58.21 变为 56.64 ms。进一步限定实际输入、输出和已读缓存长度均相同后有 1,083 对，TTFT p95 为 4,020.14 / 3,711.98 ms。闭环回放的请求构成会变化，部署判定采用上表整个窗口。

## 固定负载与正确性

在相同的 8K 输入、256 输出、C64 回放中，4K 混批两轮合并为 1,555.14 tok/s，TTFT p95 5.71 秒，TPOT p95 47.16 ms。原对齐版两次独立服务合并为 1,454.05 tok/s，TTFT p95 5.32–5.43 秒、TPOT p95 53.55–54.52 ms。全局改用 6K 后为 1,224.28 tok/s、7.51 秒、62.79 ms。固定形状的生成提速没有直接转化成 AgentX 的整机吞吐收益。

两种深度都在 8 个 rank 的 worker 日志中观察到混合前向。一个 MTP0 前向含 8,185 个 prompt token 和 7 个 decode token，一个 MTP3 前向含 4,096 个 prompt token 和 7 行四位置验证。MTP0 和 MTP3 分别在有在途用户时完成 16/16 个 65,536-token 定点检索，每种深度的 32 条生成探针都完整返回 512 个 token。MTP3 检索期间，记录到每行接受 0 / 1 / 2 / 3 个草稿的次数分别为 26 / 21 / 10 / 468。发布工作树中与目标模型、调度、MTP、compact-state 和 Mooncake 相关的 1,008 项回归在 H200 上全部通过。

## 长输入

对齐版 C90 中，TTFT ≥ 3.602 秒的 153 条请求里，119 条在减去已读缓存后仍有至少 32,768 个输入 token。基于这个分布，额外测试按实际未缓存后缀选择块长。首块沿用已有的准入预算。对确认很长的后缀，后续块可在预留所有在途验证 query 后使用剩余引擎额度，并遵守 KV 页边界与 checkpoint 切分规则。已缓存的长历史不会单独触发扩块。

同一组 16 条冷 64K 输入、32 条在途探针的独立流式测量中，普通 4K 混批的 TTFT p50/p95 为 7.30/8.75 秒，仅长后缀扩块后为 8.86/10.80 秒。两者检索均为 16/16。

选择性混批仍用 4K 块长。只有已确认未缓存后缀至少 32K 的请求符合条件。活跃长请求本来要让出本轮给 decode，而全部在途验证行和下一块 prompt 合计能装进 8K 预算时，可在同一个 rank 的一次目标前向里推进两者。普通新请求继续按周期 2 准入。[选择性混批实现](https://github.com/silencelamb/lmdeploy/commit/5558b6af44720dd824e4e1e45348dee8c5b13009)

重新独立启动后，选择性混批的 16 条冷 64K 输入获得 16/16 个正确答案，32 条在途流各交付 512 个 token。首 token 的 p50/p95 为 **6.62/7.74 秒**，普通 4K 混批为 **7.30/8.75 秒**。在 rank 0 上，两个注入波次各有 16 个连续集体轮次同时执行长输入和验证，共 30 对相邻轮次。普通 4K 混批的相同测量为 0 对。16 个相同输入的首 token 全部比普通 4K 混批更早，配对中位差为 −0.69 秒。这里比较的是定点检索的受控注入。选择性调度的发布树另通过了 359 项可在 CPU 上执行的行映射、调度、KV 分配和恢复回归。

## 复现

下列环境建立 4K 普通混批，服务端共同使用 `--max-prefill-token-num 8192`。长后缀扩块将门槛设为 32768。选择性借验证轮同时设置门槛 32768 和 `LMDEPLOY_GLM53_MIXED_LONG_OVERLAP=1`。

```bash
export LMDEPLOY_INTERLEAVED_PREFILL_TOKENS=4096
export LMDEPLOY_INTERLEAVED_DECODE_ROUNDS=1
export LMDEPLOY_DP_PREFILL_PERIOD=2
export LMDEPLOY_GLM53_MIXED_FORWARD=1
export LMDEPLOY_GLM53_MIXED_LONG_THRESHOLD=0
export LMDEPLOY_GLM53_MIXED_LONG_OVERLAP=0
```

对齐参考使用同样的部署配置并关闭 `LMDEPLOY_GLM53_MIXED_FORWARD`。正式窗使用 Weka 256K、seed 20260707、RDMA Mooncake 160 GiB/worker、每档冷启动后正式计时 1,800 秒。固定负载的工作树可启用 `LMDEPLOY_DP_PREFILL_TRACE=1`，从 worker 记录的实际 prompt token、decode 行数及验证宽度核实混合前向。正式窗关闭逐步 trace，worker 周期性报告累计混合前向数。

```bash
python benchmark/glm53_mega_moe/summarize_mixed_forward.py \
  --reference "$ALIGNED/agentx-c90" --candidate "$MIXED/agentx-c90" \
  --server-log "$MIXED/server.log"
```
