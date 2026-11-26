# GLM-5.3-Flash 固定形状 MonoKernel 源码与实验审计

本文保留历史版本的提交与测量。当前接口直接基于 main `2ff1d1f9`，已单独复测完整 AR R11 和 MTP3 target。最新口径及分支关系见[去除 #5186 后的实测与部署建议](glm53_specialized_mtp_deployment.md)。

单节点 8×H200 SXM、DP8/EP8/TP1、每 rank 八条请求，各自完成 65,536 token 的真实 prefill，再自由生成 512 步。R11 的三轮正式结果是 **12.34909 ms/步**。这个数包含 45 层、LM head、argmax 和设备状态反馈。Serving 中的排队、动态组批、流式返回和 EOS 退场在这项实验之外。

## 提交与产物身份

| 对象                     | 已核实的身份                                                                                                                                | 用途                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InternLM/lmdeploy:main` | `f3e625205470ceee1d24effd3b74d220ff7a18e5`，2026-11-25 重新 fetch                                                                           | 新接口的上游父提交，包含 A3 原生 PyTorch 后端 `6a4daf62`、ReplaySSM `12b95d12`、DP1 Graph 有效路由修复 `95f4ed4d`                                 |
| SM90 MegaMoE             | #5186 的远端 head `42dc6d3190404adf7523661d3b16eaeb642976df`，审计时 PR 开放                                                                | DeepGEMM / FlashInfer 后端。开发基线 `63ed082d` 以原提交为第二父提交，两个冲突处同时保留通信后端、MoE 选项与 ReplaySSM build context              |
| R11 发布树               | `R11-GPT6.2-Sol/release` 的 HEAD 为 `63549b282790f77ebf4bf2a0d6e44a70de14d85b`，实际发布源为 HEAD 加补丁                                    | 本轮重算 `artifacts/source-manifest.json` 中 170 个文件的 SHA-256，全部相符。单独 checkout 这个 HEAD 得到的是未打补丁的旧框架                     |
| 模型                     | `zai-org/GLM-5.3-Flash`，快照 `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`                                                                    | `config.json`、safetensors 索引和 62 个权重分片。研究与本轮接入都用同一快照                                                                       |
| 工作区 `monokernel/`     | SGLang `b959096ecf`，AMD mono `1ed489b`，NVIDIA mega `e9e01fe`，TPU megakernels `a120ca4`，DeepGEMM `44cba7d`，`vllm` 链接指向 `c2fb5c5aba` | 对照实现。AMD 包有模型 spec，NVIDIA 包有模型级 mailbox/融合链，TPU 有独立的权重打包和执行布局。R11 实际使用的 patched DeepGEMM 有独立的版本与 ABI |

上游提交、待合入的 MoE 实现和本轮接口可以按 Git 父子关系独立比较。下文以 `R` 指冻结的 `R11-GPT6.2-Sol` 目录，`D` 指其父目录，文件名均相对于这两套交付物。[上游 main 快照](https://github.com/InternLM/lmdeploy/tree/f3e625205470ceee1d24effd3b74d220ff7a18e5) · [A3 #5134](https://github.com/InternLM/lmdeploy/pull/5134) · [SM90 MegaMoE #5186](https://github.com/InternLM/lmdeploy/pull/5186)

## 一次真实 Decode 的数据流

`run_decode_profile.py` 用每个 rank、请求的独立内容分块 prefill，保存物理 block 池、状态池和 SSM cursor。各个变体恢复同一快照、第一步输入和槽位。`Glm5NextModel` 将 4096 维 embedding 扩展成四路 mHC。在每层，mHC pre 投影、归一化和 Sinkhorn 得到输入及混合系数，分别执行注意力和 FFN 前后的残差混合。最后合并四路、做 norm 和 154,880 词表投影。

- **34 层 KDA。** `Glm5NextLinearAttention` 的 Q/K/V、beta 和两级 gate 读 BF16 投影权重。QKV 投影是 `[24576,4096]`，每路 64 头 × 128。`glm53_decode/schedule.py::kda` 在副 stream 算 gate，同时主 stream 跑预打包 QKV。`kda_norm.py` 与 TileLang `kda_norm_kernel.py` 合并逐通道宽 4 卷积、门控递推和 gated RMSNorm，保留 BF16 中间舍入边界。卷积历史按状态槽保存最后三项，递推在每层每槽更新 `[64,128,128]` FP32 状态。状态读写取 indexed SSM 的 input/output IDs，batch 行号会随调度变化。
- **11 层 Sparse MLA。** 模型无 text RoPE，Q LoRA rank 1536、KV 压缩宽度 512，QK-NoPE 和 V 头宽均为 256。`schedule.sparse` 并行安排 KPool 更新/选择与 query 投影，通过吸收后的 KC/VC 权重做 BMM。先把新 latent 写入权威页，再按逻辑索引、block table 和真实 stride 转成物理地址。BF16 解码分支以 `as_strided` 取零复制读视图。`attention/sparse_mla.py` 的专用分支校验 `[8,64,512]`、BF16、int32 索引、SM90 和 `topk_length is None`。`native/decode/mla_v4.cu` 复用 FlashMLA SM90 的稀疏注意力，正式构建 `GLM53_PRE_QK=0`，按 V 分片，沿用 QK/online-softmax 顺序。动态长度参数、FP8 MLA 等走其他分支，各有独立前提。
- **KPool 和尾部。** 实验页长 64 token，逻辑上每 4 token 形成一组，与 16 个 pooled entry 对齐。每个 packed 索引页保存 128 维 FP8 key 及 FP32 scale，MLA latent 在另一个视图。完整组按 32 个 index head 打分，稳定选择最多 512 组，展开到 2048 个 token，再拼最多 3 个未成组的尾 token。负索引表示无效，后端按 kernel 要求做对齐。尾 key、尾 score 是按请求索引的 BF16 ring，MTP0 配置分配 `4+1` 行。`kpool_raw_decode_update_cuda` 消费真实 `kv_seqlens`、`q_seqlens`、`block_offsets` 和 `state_ids`。把 KV 单独迁移、重置或比较会漏掉决定后续路由与输出的状态。
- **3 层 dense FFN，42 层带共享专家的 MoE。** 每 token 对 288 个逻辑专家做带 bias 的 sigmoid top-8 选择。权重来自原始分数。`glm53_decode/r8.py::_select` 对有限分值用分数位和反向专家 ID 构造排序键，以固定 XOR 树归一化。非有限值走重复 argmax 分支。路由随真实隐藏态变化。共享专家走副 stream，路由专家及通信在主 stream 推进。等待、跨 stream 张量寿命和共享/路由结果的 BF16 加和位置都是数值契约的一部分。
- **mHC 和投影。** R7–R09 的预取、向量化 mHC、推迟到消费者的 20 次 Sinkhorn、残差 BF16 直接读取和小 M 专用 GEMM 减少发射或访存。R8 的 QKV 将 K64 半块预排成 SW128，R11 的 `r11.py::output_projection` 将 34 份 `[4096,8192]` BF16 输出权重按 TMA 消费顺序打包。split-K 边界为 `0,2752,5504,8192`，前两段的末块只读 K64，FP32 partial 按 `(p0+p1)+p2` 合并后回到 BF16。仅这份副本增加 2,281,701,376 字节/rank。正式路径仍有独立的 recurrent、投影及归并 kernel。

## 专家并行与进展条件

`glm53_mega_moe.py` 校验本地 BF16 `[8,4096]`、FP32 路由权重、int64 专家 ID、E4M3 FC1/FC2 与 FP32 block-128 scale，要求原有连续专家所有权。每 rank 原有 36 个专家，`replica.py::make_weights` 在准备阶段以 NCCL P2P 获得 18 份副本。8 个 rank 共保有 432 个物理专家槽，逻辑专家仍为 288。按实际路由位图选择本步 owner，原专家内累加顺序和 top-8 槽序保持一致。

`r10.py::_Planner` 在 capture 之前为 EP group 建立 symmetric memory 和 peer pointers。准备核交换带代数的位图/原始槽序并写本地连续 pull 任务表。每个消费者等待对应源 rank 的就绪信号。`routed_r11.cu` 以 132 CTA、384 线程调度 patched DeepGEMM 的完整 routed body。M32、K-major FP8、WGMMA 和 TMA stage 释放次序来自 R6。R5 返回区以成对 BF16 NaN 位型 `0xffffffff` 为 sentinel，依赖真实有限输出、单消费者重置、代际顺序及所有 peer 到齐。系统作用域 release/acquire 发布跨卡工作，局部 fence 和 barrier 处理同卡 shared-memory 可见性。共享专家和上一步的缓冲都必须在复用前完成。`r10.py` 还写死了 432 个物理槽、128 token/rank 的池容量及 `15104×4096` 激活布局。容量字段不能用作可支持的实时 batch 上限。

单节点 NVSwitch 上的压力回放涉及负载偏斜、拥挤路由与连续重放。若某 rank 改走 DeepEP、使用另一代 workspace、跳过某层或缺席，另一端可能永久等信号。原实验对不合规的部分输入直接报错，某些层有局部 fallback。这些分支没有形成任意混合 batch 下的全组协议。新接口在进入 forward 前协商。最小投影示例复用 LMDeploy 的 MoE 路径，完整 `FullR11` provider 已接入这里的 132-CTA 专家协议。

## Shape、图和正确性的边界

| 结论                       | 证据与适用范围                                                                                                                                                                                                                                                    |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **历史整模已验证**         | 单节点 8×H200，DP8/EP8/TP1，每 rank 恰好 8 个真实 decode token、每请求 q=1，固定槽/预分配页。原始 64K prefill 后连续 512 步，自回归产生全部 32,768 个 token。每行历史长度逐步从 65,536 增长。固定 batch 和内存几何并不冻结逻辑长度                                |
| **局部长度能力已实现**     | KPool score 与 radix Top-K 读每行有效长度，短行填 `-1`，R09 的 `topk_r09.cu` 对大于 17,408 **组**的行转 leader CTA 顺序算法。该数是算法分支点。稀疏 MLA 读物理页映射和掩码。真实快照及局部变长/ties 测试见 R09 的 `reports/topk_oracle.json` 与 `mla_oracle.json` |
| **冻结研究版本的验收边界** | 不同 rank 的真实 B、dummy、slot 回收、混合 phase、动态页重用、连接器迁移、任意 1M 上下文、在线换权和 NPU。模型配置的最大位置数 1,048,576 只描述模型。原有图依赖预热后的地址、打包权重与固定通信组                                                                 |
| **已运行的研究变体**       | 512-CTA cooperative KDA 实际做了 QKV/gate/Sinkhorn/卷积/递推/输出。与 42 个 cooperative MoE 合用得到 13.20317 ms/步。无损 FP8 编解码能还原原字节，提前展开版整模 18.17616 ms。PDL、L2 预取及更大融合区分别有微测和整模消融，未进入正式路径                        |
| **尚未实现**               | 把 45 层所有算术变成一个常驻 kernel。整条 R11 的 target verify、draft、按接受数提交/回滚，以及任意负载的专家协议。A3 上的该算法也尚无实现                                                                                                                         |

`whole_step.py` 预热模型，捕获 forward、head、argmax、token/history/SSM 反馈与记录，捕获期间临时将内层 graph replay 展开为 prepared callable。`native/whole_step/device_loop.cu` 的计时标记使用 `%globaltimer`，finish 节点按步数做 device graph tail launch。GPU 自循环中的 host 标量、页表容量和请求集均预先确定。实际 Serving 由 scheduler 决定下一步成员，因而本轮接口只接管一次模型 forward。研究用 `r11_source.install` 及命令队列可在同一进程切换实验开关，正式服务也需要独立的默认图。

## 性能、误差与成本

| 配对来源        | 实现                              |                              ms/步 | 口径                                                     |
| --------------- | --------------------------------- | ---------------------------------: | -------------------------------------------------------- |
| R09 源码实验    | 原始 eager / 框架 Full Graph      |               101.70259 / 18.64388 | 分别从框架入口测量，每轮抽 4 步全词表                    |
| R11 source      | 原始算术的 GPU 循环               |                           18.01130 | 全 512 步词表参考，同一队列                              |
| R11 source 三轮 | R10 / R11 紧凑投影 / 再加 132 CTA | 12.37926 / 12.35353 / **12.34909** | 每步先取八个 rank 的最大 GPU body 时间，再跨步、跨轮平均 |
| R11 recheck     | 重编 R11 两轮                     |                12.34649 / 12.34497 | 重新真实 prefill，原生 BF16 logits 字节复验              |

三轮 R11 的 p50/p95/p99 均值为 12.4350/12.8246/12.9128 ms。含回环为 12.351795 ms/步，固定 64 请求约 5,181 tok/s。相对同队列 R10 减少约 30.17 µs，其中紧凑投影约贡献 25.73 µs。`validation.json` 的 release 队列有 96 个带参考的 rank-run，独立 raw recheck 有 32 个。后者明确记录原生 BF16。除 token 和 logits，harness 还检查四类语义状态、完整物理池及 cursor。研究队列存在未完成变体，正式 release/recheck 的 incomplete 数均为零。

`bytes-budget.json` 将逻辑关键路径载荷估为 35.54 GB/步。以单独测得的连续读 4.574 TB/s 折算约 7.77 ms，这是带宽参照。L2 命中、中间写回、同步与阶段重叠使其不能直接给出可回收的毫秒数。R11 插桩下 MoE、KDA、MLA 分别约 5.61、3.43、1.91 ms，探针本身约增加 0.389 ms。`raw-measurements.json` 中 R11 进程 allocated 约 129.83 GB/rank，含先前方案在同一进程留下的资源。服务的 KV 可用容量应在打包权重完成后重新规划。

## 冻结实验的复现与交付清单

| 位置                                                                                                                  | 内容 / 依赖方向                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `R/release/lmdeploy/pytorch/models/glm5_next.py`、`backends/cuda/{kda,kpool,attention/sparse_mla,moe/blocked_fp8}.py` | 模型及算子分派，调用 `kernels/cuda/glm53_decode`、`glm53_mega_moe.py`、mHC kernel                                                                                                                         |
| `R/release/benchmark/glm53_monokernel`                                                                                | 源码发布说明，固定形状 driver、分轮队列、构建器、`native/decode` 的 C++/CUDA/PTX 及 `native/whole_step` 的设备循环                                                                                        |
| `D/r5/release_decode`、`D/r6/release_vendor`、`D/r7/release_decode`、`D/r8/release_native_v3`                         | R5 稀疏/原生 helper、R6 独立 DeepGEMM Python 扩展和头文件、R7 mHC、R8 QKV 与 planner。`D/vendor/whole_step` 提供设备循环库                                                                                |
| `D/R09-GPT6.2-Sol/native/release_v2`、`D/R10-GPT6.2-Sol/native/release`、`R/native/release`                           | R09 的 Top-K/mHC 等库、R10 的 prepare 和 paired8 头、R11 的紧凑 cubin/host launcher/132 CTA expert                                                                                                        |
| `R/scripts`、`R/build`、`R/results`、`R/reports`                                                                      | 构建、启动、实验迭代。正式数据在 `results/release`、`results/raw_recheck`，`reports/source-measurements.json`、`raw-measurements.json`、`graph-audit.json`、`validation.json`、`stages.json` 可以交叉核对 |
| `R/artifacts`                                                                                                         | 补丁 SHA-256 `eea660ad0a14067255e901abdc6a7da457774882019f80455613e5e2a45edd1f`。归档 SHA-256 `32a9182e62aef9f63c8a2705979edd9e95280ca3832015ad0d02c257b9e00b8b`。本轮也重算了这两个文件的摘要            |

冻结环境为 Python 3.12、PyTorch 2.12.1+cu130、Triton 3.7.1、CUDA 13.0、GCC 13.3，另用 TileLang、DeepEP/NVSHMEM、FlashMLA/CUTLASS 与 libelf/libdw 开发头。构建器固定 DeepGEMM `724a799bd04382732df8ba73611b4a2702879ac6`、FlashMLA `9241ae3ef9bac614dd25e45e507e089f888280e0`、CUTLASS `147f5673d0c1c3dcf66f78d677fd647e4a020219`。R6 扩展在自己的 package 下安装，manifest 同时记录精度、布局、返回与 stage ABI。R11 构建显式生成 `sm_90a`，复用 R10 paired8 和 R6 的头文件。`scripts/launch.sh` 给出了全部原生目录与 `PYTHONPATH` 优先级。只复制 R11 的三个二进制无法复现完整实验。

在已有配套容器和冻结前置产物的机器上，从宿主机的 `R` 目录运行以下命令。容器挂载 `D` 为 `/work`、checkpoint 仓库为 `/model`。选择新的结果目录，保留原对照。

```bash
python3 release/benchmark/glm53_monokernel/prepare_suite_r11.py results/reproduce-v1
bash scripts/launch.sh --model /model/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a \
  --output /work/R11-GPT6.2-Sol/results/reproduce-v1
python3 ../r6/scripts/audit_results.py results/reproduce-v1
python3 ../r8/scripts/measurements.py results/reproduce-v1 reports/reproduce-v1.json
```

`release/benchmark/glm53_monokernel/README.md` 按 R4–R11 给出从源重建命令，`build_r11.py --r10-include ... --provider-include ...` 生成末级 `manifest.json`。研究 archive/模型权重由持有者另行分发。当前 PR 随附的小范围实现可在独立 LMDeploy checkout 构建并运行，详见其接入与测试入口。[本轮接口设计](specialized_execution.md) · [接入示例](../../../examples/specialized_execution/README.md)

完整 R11 在新版引擎中的验证使用新的分支与独立结果目录，含 64K×512、全词表与缓存对照、真实客户端和路由计数。[新版集成实测](glm53_full_r11_results.md)
