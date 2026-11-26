# GLM-5.3 的专用 decode、MTP3 验证与 AgentX 上线选择

GLM-5.3-Flash 自带的 NextN predictor 及原生 MTP 已在上游 main。本文的八卡实验验证本轮新增的 target 专用执行入口。各阶段的职责、参考实现和 C70 计时口径见[从 draft 到接受前缀](glm53_mtp_execution_flow.md)。

## main 与 MegaMoE 的关系

接口只接管一次 target forward。PR 的父提交是 LMDeploy main `2ff1d1f97c855e795ad91336b69cf97113fa7bab`，包含 A3 后端 `6a4daf62`。#5186 的 `42dc6d31` 是另一条分支，提供 SM90 的可选 MegaMoE 后端。接口不导入它，也不要求设置 MoE backend。加载器、输入、状态槽和输出契约使用引擎已有类型。当前接入点只启用 PyTorch CUDA target，A3 继续使用原后端。[接口](../../../lmdeploy/pytorch/specialized_execution.py) · [A3 提交](https://github.com/InternLM/lmdeploy/commit/6a4daf62) · [SM90 MegaMoE PR](https://github.com/InternLM/lmdeploy/pull/5186)

- **完整 R11** 自带冻结的 R6 DeepGEMM、专家权重打包、八卡通信和 132-CTA 程序。它需要自己的运行包，普通回退使用现有 DeepEP 权重布局。若部署选择 #5186 的专用权重格式，FullR11 声明空覆盖。这里的两个专家实现分别拥有同一层的分发、计算和归并。
- **局部专用计算** 可以沿用部署选定的 MoE。投影 provider 创建只供自己 forward 使用的子模块视图，借用同一份原模型参数和 cache，只换 34 个 KDA 输出投影。其他层和未命中的 forward 继续使用原实现，主仓无新增 GLM 模型钩子。
- **共存实测**在隔离的 checkout 单独合并 #5186。配置和 Ray 有重叠改动，手动保留两个开关。用 `deepgemm_mega_moe` 的原生 MTP3 为同进程参照，在八卡的真实 B2×Q4 verify 中，rank 0 的 GPU profiler 同时计到 34 次 R11 输出投影、42 次 staging 和 **42 次 SM90 DeepGEMM MegaMoE 主核**。同快照的 logits、候选/提交状态与回退续跑全部通过。该对照衡量相同 MoE 后端上两种 target 的差异。[投影实现](../../../examples/specialized_execution/glm53_r11.py) · [完整 R11 专家实现](../../../examples/specialized_execution/glm53_full_r11/experts.py) · [逐项数据](../../../examples/specialized_execution/mtp_main_validation.json)

## 去除 #5186 后的完整 R11

模型快照 `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`，8×H200、TP1/DP8/EP8，每卡 8 个请求、64K prefill、32 步 warmup 和 512 步自由解码。计时从 GPU 记录的每步起止取八卡最大值，再对 512 步取平均，含 45 层、LM head、argmax、反馈和全词表记录。先运行真实 prefill，再恢复原研究快照；本次八卡的首次输出和五个物理缓冲均与新 prefill 相同。各条件从同一状态开始。[复现实验与产物](../../../examples/specialized_execution/FULL_R11.md)

| 路径                              |                     平均 ms/步 |
| --------------------------------- | -----------------------------: |
| 普通 target，设备循环             |                       17.99189 |
| 完整 R11，设备循环第 1 / 2 / 3 轮 | 12.34031 / 12.34168 / 12.34127 |
| 完整 R11，三轮平均                |                   **12.34109** |
| 同一步图，主机连续提交 replay     |                       12.34515 |
| 原冻结 R11 报告                   |                       12.34909 |

八卡每步时间戳、token、logits/状态摘要和 GPU 图归档在研究分支，可解包后用 `analyze_full_r11.py --device no5186-r11` 重新聚合。[本次完整原始数据](../../../examples/specialized_execution/r11_main_64k_512_evidence.tar.gz)

每条八卡轨迹的 32,768 个 token 与冻结结果相同。对照普通 target 的每步全词表 BF16 logits、四种语义状态、五个物理池/游标的字节摘要也相同。17.958 ms 来自另一项实验，只替换 KDA 输出投影，其余计算沿用同一 MegaMoE。此前完整 R11 的真实 64 路、64K HTTP 测试在预捕获后，稳态 GPU start-to-start 平均 13.624 ms，p50 13.272、p95 13.658；这组历史 Serving 读数使用旧的组合分支。真实路由的关键 rank 唯一专家权重载荷从 19.007 增至 22.866 GB/步，profile 中 42 个专家 grid 合计增加 0.825 ms。[历史 Serving profile 与时间边界](glm53_full_r11_results.md) · [冻结源码与构建审计](glm53_monokernel_audit.md)

## 合作方如何交付 MTP3 计划

1. 在 `describe(ctx)` 检查真实的模型、设备、权重形状与部署深度。返回 `DecodePlan('verify-b2-d3', batch_size=2, max_kv_len=66048, graph_safe=True, num_spec_tokens=3)` 时，承诺每个参与 rank 有两条真实请求，每条有四个 target 输入位置。历史长度、页表、各请求槽号可以不同。只支持固定历史的内核应在能证明此前提时才注册。
2. 在 `run` 消费 target 本来的 prepared kwargs。按每个候选位置写入引擎提供的 KV 和辅助状态，返回原 target 的所有位置 hidden 和可选辅助输出。保持普通模型与 cache 可直接执行。普通 AR B8×Q1 和 verify B2×Q4 都有八行向量，只有逐行无状态的运算可以直接复用这种展平。
3. 引擎的 draft 继续使用自己的模型、cache、graph 和 proposer。普通 rejection sampler 计算每请求接受的连续草稿数 a。引擎提交本轮第 1+a 个输入后的状态，下一轮从该位置续写。拒绝后尚在物理页中的暂写数据须受逻辑长度和状态 ID 隔离。普通/专用图有不同的 key，缺一个 rank、不同的实际 B/Q、ragged、dummy、混合 phase 或超界时，全组回到普通 target。已经开始写状态的 forward 发生异常会传播给引擎。
4. 准备、按组协商、graph 重绑、权重更新和资源释放遵守同一套生命周期。API 数据结构只描述执行和状态语义。扩展到 NPU 时还需对接该后端的 runner、资源和图实现。

`glm53_r11:MTP3Projection` 对 GLM-5.3 的 34 个真实 KDA 层执行现有 sm_90a 原生投影。完整 R11 本轮仍只覆盖 B8×Q1，其递推、卷积、KPool 和 EP workspace 均需支持每个请求的多位置输入及提交边界，才能声明 B8×Q4。draft 加速、ragged verify、ReplaySSM verify、A3 真机执行均在本轮实测范围之外。[合作方接口说明](../../../examples/specialized_execution/README.md) · [接口数据结构](../../../lmdeploy/pytorch/specialized_execution.py) · [引擎内的状态提交](../../../lmdeploy/pytorch/model_inputs.py)

## 复现 MTP 正确性

从本研究分支根目录运行。按 [R11 原生包说明](../../../examples/specialized_execution/FULL_R11.md) 设置 `MODEL`、`KIT`、`PYTHONPATH`。八卡示例需要本机的 CUDA 13、PyTorch 2.12.1+cu130、Triton 3.7.1、LMDeploy 及默认 DeepEP。CPU mock 使用可导入 LMDeploy PyTorch 模块的环境。

```bash
export PYTHONPATH="$PWD:$PWD/examples/specialized_execution:$KIT${PYTHONPATH:+:$PYTHONPATH}"
export GLM53_R11_PROJECTION_DIR="$KIT/glm53_r11_artifact"
export LMDEPLOY_DP_MASTER_ADDR=127.0.0.1
export LMDEPLOY_DP_MASTER_PORT=24930
python examples/specialized_execution/mock_mtp.py
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_mtp3_states.py \
  "$MODEL" results/graph --profile
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_mtp3_states.py \
  "$MODEL" results/eager --eager --profile
python examples/specialized_execution/verify_mtp_serving.py "$MODEL" results/stock
python examples/specialized_execution/verify_mtp_serving.py "$MODEL" results/specialized \
  --provider glm53_r11:MTP3Projection --reference results/stock/tokens.json
python examples/specialized_execution/verify_mtp_serving.py "$MODEL" results/live-shadow \
  --provider glm53_r11:ShadowMTP3Projection --eager
```

- **CPU mock** 使用同一个计划选择器、显式四个候选槽和原生 PyTorch greedy 拒绝采样。接受 0/1/2/3 个草稿分别提交 1/2/3/4 个输入，下一步的普通执行与逐 token 参照相同。

- **真实全模型，在同一状态分叉**。各 rank 的两条请求先分别 prefill 63/65 个 token，物理页交错并跨过 64-token 边界。为保证四种接受情况均出现，按普通 target 的确切预测构造匹配/不匹配的草稿。每种情况分别恢复相同快照，运行真正的专用/普通 target、GPU rejection sampler 和引擎的 cursor 推进。只有 rank 0 将 B2 改为 B1，八卡全部退回后再执行一步。Eager 和 Full CUDA Graph 都在所有 rank 比较 hidden、全词表 logits、发布 token、提交槽、历史长度和五个物理缓冲/游标。两条分支的比较逐字节通过。每种运行另有一次独立 profile 核实 34 次原生 launch。构造的草稿仅用于覆盖边界。

- **真实 HTTP** 使用原生 MTP proposer、调度、采样和输出路径。每次 1 个单请求加 16 个并发请求，共 1,544 个生成 token。用相同 runtime 和显式 greedy 配置进行三次独立启动。原生 A/A 的 17 个完整请求中 12 个逐 token 相等；原生 B 与专用 verify 有 13 个相等。更早的独立冷启动中，单请求第一个 prefill 输出已经分叉（492 与 726），该形状不会选中专用计划。脚本对 `--reference` 严格断言并保留两侧 token；这些独立启动的全输出一致性检查未通过。确切的跨启动分叉来源尚未确定。部署判断需同时参考同输入校验、内容质量与真实工作负载。

- **同一次 HTTP 请求的逐层数值对照**。在 eager 诊断 provider 内，对每个命中步骤的每个 KDA 输出投影，把同一份真实输入同时交给普通投影与 R11 原生投影，只推进一份模型状态。八卡各 23 轮，共 184 个 rank-forward、6,256 次逐层比较；204,996,608 个 BF16 元素的 bit pattern 全部相等，最大绝对差为 0。独立无额外计算的 Graph 测试同样在八个 worker 各记录 23 次真实 replay。Shadow 会额外运行投影并同步，仅用于数值诊断。[八卡原始摘要、合成请求 token 和逐层统计](../../../examples/specialized_execution/mtp_verification_samples.json.gz) · [分阶段结果](../../../examples/specialized_execution/mtp_main_validation.json)

接口及相邻模块的回归测试 **808 个通过**，包括真实 TP2/TP8、DCP1/DCP2 的 Ray symmetric LM-head gather、MTP 和 Ascend host 测试。一个依赖昇腾硬件运行时的 import 用例跳过。全仓 pre-commit 通过。

要复核可选 #5186 组合，可在单独的临时 checkout 中从历史组合提交 `63ed082d` 合并 main `2ff1d1f9`，再合并接口 `f86378ac`。对 `lmdeploy/cli/serve.py`、`lmdeploy/messages.py`、`lmdeploy/pytorch/config.py`、`lmdeploy/pytorch/engine/executor/ray_executor.py`、`lmdeploy/serve/openai/launch_server.py` 的文本冲突，使用保存在 `1b5d3c6a` 的组合开关版本。其余采用 Git 的自动合并。实测 checkout 的 `lmdeploy/` tree 为 `ef5c8aa98176219ec42111a2bee193299991e918`。在该环境将 #5186 的 SM90 DeepGEMM 放在 Python 搜索路径，给同一份 `check_mtp3_states.py` 增加 `--moe-backend deepgemm_mega_moe --profile`。对照时普通与专用 target 都使用相同的 MegaMoE。[历史组合提交及源文件](https://github.com/silencelamb/lmdeploy/tree/1b5d3c6adf75786258c230bac72ed7832bb3ccd9) · [可选后端构建](https://github.com/silencelamb/lmdeploy/blob/d16aabc6fef030419745db1cb5d7d4204a64e735/docs/zh_cn/benchmark/glm53_sm90_megamoe.md)

## AgentX 应先优化哪条路径

**以现有 MTP3 + MegaMoE 为 AgentX 生产基线，先优化它的 target verify。** 同时保留完整 R11 AR，作为固定长上下文、低接受率或明确禁用投机的独立候选。这样可以用同一套接口做两个可独立回退的计划。MTP3 的每卡 B8 包含 32 个 target 输入位置。先用真实 C70 轨迹统计可覆盖的 B/Q、KV 长度和各 rank 的最低覆盖，再选择要生成的形状。8 行投影样例验证了接线和状态契约，B2×Q4 在生产上的覆盖和加速仍需另测。

**固定步时和线上每 token 时间有不同的计时边界。** 可复核的 Weka 256K、Mooncake、seed 20260707 的 C70、1,800 秒窗口报告了 2,307 个完成请求、1,285.22 输出 token/s、0.97663 个满足联合 SLO 的请求/s。该轮启用 DeepGEMM MegaMoE 和 FlashKDA prefill。按输出 token 间隔加权，TPOT 为 **16.617 ms/token**。各请求等权时，均值为 20.141 ms、p50 15.189 ms、p95 47.566 ms。缓存命中口径的已读取 token 比例为 92.24%。C32 的另一份同深度实验中，原生 MTP3 与 DeepGEMM MTP3 分别为 966.51 和 1,052.21 输出 token/s。C32 和 C70 的缓存、seed 与负载条件各有差异。[C70 配置和结果](https://github.com/silencelamb/lmdeploy/blob/d16aabc6fef030419745db1cb5d7d4204a64e735/benchmark/glm53_mega_moe/results/qa_agentx_mtp3_mooncake.json) · [C32 对照](https://github.com/silencelamb/lmdeploy/blob/d16aabc6fef030419745db1cb5d7d4204a64e735/benchmark/glm53_mega_moe/results/h200_megamoe.json)

C70 的服务计数是 760,654 请求轮、2,281,962 个草稿位置、1,560,069 个接受。平均每请求轮逻辑上推进 `1 + 1,560,069 / 760,654 = 3.051` 个 token。EOS、最大输出和客户端截断应使用实际交付的 token 数核算。对同一可比请求群，若整轮暴露时间为 T、平均推进数为 A，则每 token 计算成本约为 T/A。T 应覆盖 draft、target verify、拒绝/提交和暴露的调度/通信间隙，重叠部分只算一次。假设接受率和组批相同、所有轮命中，每轮 verify 节省 1 ms 约折合 0.328 ms/token。若只命中部分轮，按命中轮的请求量、实际产出和关键路径加权。多路服务吞吐用全局输出 token 除墙钟时长，逐请求同时看 TTFT、TPOT 和 goodput。

**先复用现有专家权重与通信。** 在部署统计中分别标记 draft、verify、提交、prefill 干扰和空转，观察 KDA/MLA/KPool、mHC、小 GEMM 与相邻边界实际暴露了多少时间。优先实现能在高频形状下重用权重、保留所有候选状态的 target 子图；若 profile 指向专家，单独比较或替换该层的 expert 程序。全量移植 B8×Q1 R11 的代价很具体：只计算 42 层、每 rank 54 个专家的额外 FP8 权重就是 `42×54×3×4096×2048 = 57.08 GB`/rank，尚有 scale、QKV 等派生权重；原权重还要保留供回退。缓存读比例很高的 AgentX 工作负载需要同时记录实际可用 KV blocks、prefix hit、重算和并发上限。[打包代码](../../../examples/specialized_execution/glm53_full_r11/experts.py)

**上线判定采用四组同负载对照。** 依次测 stock AR、完整 R11 AR、MTP3 + MegaMoE、同一 MTP3 + MegaMoE 加 target provider。固定 checkpoint、数据及 seed、到达与并发、KV 格式、prefix/Mooncake 启动条件、输出上限、采样和 SLO；交错重复，在同一设备预算下同时记录可用容量。还可加一组限定相同 KV 容量以区分纯计算收益和显存代价。分别记录每 rank 的 B/Q/命中、接受前缀分布、实际输出数、各阶段 GPU 与暴露墙钟时长、TTFT/TPOT 尾部、tok/s、SLO goodput、数值和回退后的状态。只有在代表性负载下提高 goodput 或缩短延时且守住正确性及容量目标，才应启用相应 provider。本次完成可组合性与正确性验证，新增 provider 的 C70 增益尚待这组实测。
