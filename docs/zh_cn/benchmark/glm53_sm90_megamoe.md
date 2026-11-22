# GLM-5.3-Flash 在 H200 上的 MegaMoE 实测

单节点 8×H200，DP8 + EP8 + TP1。GLM-5.3-Flash checkpoint 固定到 `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`，LMDeploy 基线为 `63549b28`，Torch 2.12.1+cu130。本页对照保留 BF16 的 KDA prefill，在相同 MTP 深度分别比较原生 DeepEP 和 DeepGEMM MegaMoE。[复现入口](../../../benchmark/glm53_mega_moe/README.md) · [结构化测量结果](../../../benchmark/glm53_mega_moe/results/h200_megamoe.json)

## 实现

`--moe-backend` 可选 `default`、`deepgemm_mega_moe`、`flashinfer_mega_moe`。本次 DeepGEMM 候选使用 interleaved FC1 和 shared expert 融合。EP rank 按同一组 DP token counts 选择本步通信路径及 workspace。设置 `LMDEPLOY_GLM53_MEGA_PREFILL_CAPACITY=8192` 后，短 prefill 和不同 rank 处于 prefill/decode 的混合步也可以运行融合核。纯 decode 时利用 CUDA Graph，以无效 expert ID 屏蔽 padding 和空 rank。MTP 的 target 验证使用所选 MoE 后端，draft 保持原生路径。

本 checkpoint 的 gate 和 up 各有独立 FP32 scale。interleaved 权重布局避免每 rank 多存 23.625 GiB 的 FC1 副本，prefill 的 grouped GEMM 直接读交错布局。FlashInfer 要精确保留两组 scale，需要应用随附的 split-scale 扩展。模型继续使用 FP32 noaux 路由、2.5 倍 routed scale 与 shared expert。[后端与依赖补丁](../../../benchmark/glm53_mega_moe/README.md)

## 同机固定输入输出

每个 rank 配置 8,192-token prefill 预算、32 个请求槽，关闭 MTP。使用相同输入 token ID、滚动并发、按当前在途请求数分流，每档 4 倍并发数的请求。LMDeploy 每 rank 49,152 个物理 KV 块，与 vLLM 均约 314 万可用 KV tokens。vLLM 参照为同机合并提交 `5328ae2fef`。吞吐取全部有效输出 token 数除以窗口墙钟时间。

| 输入 / 输出、总并发 | LMDeploy 原生 tok/s | LMDeploy MegaMoE tok/s | vLLM 默认 tok/s | vLLM MegaMoE tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1K/1K，64 | 3,309.71 | 3,620.81 | 3,562.50 | 3,911.65 |
| 8K/256，16 | 779.55 | 876.61 | 803.30 | 933.17 |
| 8K/256，64 | 1,316.91 | 1,603.32 | 1,492.19 | 1,940.19 |

8K/256、C=64 下，LMDeploy 的同配置吞吐提高 21.7%。同一台机器上的滚动请求对照也显示，长输入补入时首 token 前等待显著影响整机结果。

## AgentX 长窗与 MTP

Weka 256K 公开 trace，C=32，seed 42，每 lane 预热 10 条，每组一个 1,800 秒正式窗口。这组使用本地 prefix cache 和 sticky-least-loaded 路由。各组通过调节 KV 块数对齐约 125,451 MiB/rank 的就绪驻留预算。逐请求 SLO 是 TTFT 3 秒、每用户输出 40 tok/s。

| MTP 深度 | 原生 tok/s | MegaMoE tok/s | 同深度增幅 | 原生 → MegaMoE TPOT p50 | 原生 → MegaMoE 达标请求数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 575.86 | 645.22 | 12.05% | 39.37 → 34.36 ms | 330 → 460 |
| 3 | 966.51 | 1,052.21 | 8.87% | 17.83 → 15.01 ms | 1,424 → 1,698 |
| 5 | 1,045.56 | 1,096.64 | 4.89% | 16.12 → 13.77 ms | 1,579 → 1,796 |

MTP3 和 MTP5 的 MegaMoE 候选在每次验证中平均可提交 3.079 / 3.769 个 token。在相近驻留显存下，MTP3 保有 3,178,624 个可用 KV tokens/rank，MTP5 为 2,331,456。跨深度增加 4.22% 全机输出吞吐的同时，每 rank 可用 KV 减少 847,168 tokens。MTP3 因而是高并发、长上下文部署值得优先评估的深度。闭环轨迹会随执行速度推进到不同回合，结构化结果也保留各组请求数、缓存命中率及客户端错误计数。

## 固定形状和数值边界

单 rank 先做 8 条真实的 65,536-token prefill，再沿相同 token 历史运行 512 步。prefill 使用原生路径时，原生、DeepGEMM 和 FlashInfer 的整步 decode 约为 18.66、17.24、18.48 ms，以三轮中最慢 rank 的轮均值计量。与原生对照，DeepGEMM 和 FlashInfer 的 argmax 匹配约 97.72% 和 97.78%。42 个 routed MoE 层之外，模型还有线性投影、KDA、MLA 和 KPool，它们共同决定整步耗时。

开启融合 prefill 后，整组 64K prefill 的 rank 中位时间从 53.58 秒降至 47.40 秒。64 条输入有 32 条首 token 匹配原生，沿参考轨迹的 decode argmax 匹配 97.26%。最大 rank 的采样 logits 相对 L2 为 19.16%。完整 GSM8K 的配对 5-shot greedy 自由生成中，保留原生 prefill 时答对 1,219/1,319，启用融合 prefill 时答对 1,216/1,319，差 −0.23 个百分点，配对 bootstrap 95% 区间为 [−1.36, +0.91] 个百分点。需要逐 token 复现的部署应使用已验证的参考路径。固定形状工具将 KV 存储字节与 MLA/KPool 数值采样分别报告。
