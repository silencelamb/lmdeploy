# GLM-5.3-Flash 部署对照：MTP3、Mooncake 和 AgentX

这组实验考察当前组合部署的负载边界。目标模型运行 DeepGEMM MegaMoE，开启融合 prefill 和 shared expert，KDA prefill 使用另一个执行优化分支的 FlashKDA。单独衡量 MegaMoE 收益的同深度对照见[基础测量](glm53_sm90_megamoe.md)。本次测试的模型运行时代码可从[执行分支的固定提交](https://github.com/silencelamb/lmdeploy/tree/7b28f6db1b0be75fce186e8522a82bde306d4d97)复核。

## 负载与判定

同一个 Weka 256K 公开数据集、InferenceX AgentX MVP 场景、seed 20260707、流式 chat、服务端 token 计数，每档 1,800 秒。总并发从 70 起步，每次增加 10。每档重新启动服务与 Mooncake Store，使用 MVP 场景的默认轨迹预热。只有完整有效的正式窗口进入比较。

并发升级使用两项中位数，TTFT p50 < 3,000 ms 且 TPOT p50 < 25 ms。TPOT 取 AIPerf 的逐请求 `inter_token_latency.p50`，其分子是首 token 之后的生成耗时。QA 的逐请求 goodput 使用 TTFT 3,000 ms 和单用户输出速率 40 tok/s，同时记录 goodput 目标 0.0074468 req/s。goodput 和达标率与两项中位数分别报告。

## 服务配置

- 单节点 8×H200，DP8 + EP8 + TP1，MTP3。模型 snapshot `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`，block FP8 权重、BF16 激活与 MLA KV。
- 目标模型使用 `--moe-backend deepgemm_mega_moe`、interleaved FC1、fused shared expert、8,192-token MegaMoE prefill 容量。每 rank 32 个请求槽、8,192-token prefill 预算，session 上限 524,288，KV 块长 64，自动缓存系数 0.7。
- 启用本地 prefix cache 和 `MooncakeStoreConnector` 的 `kv_both` 角色。每 rank 32 个 prefix checkpoint，prefill 对齐 64、只保存末个 checkpoint，decode 首步保存、后续间隔 1,024。router 使用 sticky-least-loaded。
- Mooncake 为嵌入式 RDMA Store，`P2PHANDSHAKE` 元数据，每 rank 160 GiB 全局 segment、4 GiB 本地缓冲，全机 1.25 TiB store，关闭 SSD offload。启动后先在一个 worker 预填充，再到此前未持有该前缀的 worker 读取，核对外部命中计数和响应中的 cached tokens。

与历史 QA 对齐的是客户端场景、数据、种子、窗口与评估目标。旧 QA 表给出 LMDeploy 与 vLLM 0.33.0 的汇总，未附双方完整服务端命令、显存配额及硬件快照。历史表列出 ITL 平均值和 p99、单用户速率 p50；其中没有可直接比较的 TPOT p50。跨部署差值包含上述实现及配置差异。

以下为可移植的客户端参数。服务端需使用与集群 NIC 相符的 RDMA 配置，`WORKER_METRICS` 是八个 worker 的监控 URL。

```bash
aiperf profile --scenario inferencex-agentx-mvp --url "$ROUTER_URL" --model glm53-flash \
  --endpoint-type chat --endpoint /v1/chat/completions --streaming --use-server-token-count \
  --tokenizer zai-org/GLM-5.3 --tokenizer-revision e0b07fd2751b42d5efa199cc02c2b271deadc516 \
  --public-dataset semianalysis_cc_traces_weka_062126_256k --max-context-length 256000 \
  --random-seed 20260707 --concurrency "$C" --benchmark-duration 1800 --ui simple \
  --goodput 'time_to_first_token:3000 output_token_throughput_per_user:40' \
  --extra-inputs ignore_eos:true --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 1 --artifact-dir "$RUN" \
  --server-metrics "$ROUTER_METRICS" "${WORKER_METRICS[@]}"
python benchmark/glm53_mega_moe/agentx_slo.py "$RUN/profile_export_aiperf.json"
```

评分工具在两项中位数均通过时返回 0，在有效窗口首次越过阈值时返回 10。不可用的提交或错误的正式窗口长度会单独报错。[评分实现](../../../benchmark/glm53_mega_moe/agentx_slo.py)
