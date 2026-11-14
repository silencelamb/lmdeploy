# PR 顺序与 decode monokernel 设计建议

2026-11-14。本文是基于当前代码与实测边界的建议，未实现新的 megakernel，也没有测得其性能收益；只在 fork 文档分支维护。

同事的其他 Ascend 功能按 [接入清单](colleague_prs.md) 分别适配到当前原生后端；rebase 用于同步基线，功能迁移另做明确提交。

## 迁移 PR 与同事 Mooncake PR

**建议先提交不含 Mooncake 的迁移 PR，无须等 #5030。** 原生 Ascend 迁移的 `OpsBackend`、`BuildSpec`、`GraphRunner`、算子和模型适配不依赖 Store 的设备传输层。当前 `dev/ascend-a3-native` 正好保留这个边界。

[InternLM/lmdeploy#5030](https://github.com/InternLM/lmdeploy/pull/5030) 在本次查询时仍是 open draft、未合并；本机验证固定在 `1a09270b0835bb871be4eb4df70f9d04c8ccaec2`。适配由同事 wanfengcxz 提供，来源与合并处理在集成提交中保留。状态快照见 [记录](evidence/pr5030-status-20261114.json)。

| 目标 | 推荐分支/提交方式 |
| --- | --- |
| 上游审查 Ascend 去 dlinfer 迁移 | 从 `dev/ascend-a3-native` 提 PR，描述已测范围与非阻塞观察 |
| 保持 Mooncake 可用供开发验证 | fork 的 `codex/ascend-a3-native-mooncake` 独立保留 |
| 上游交付 Mooncake 集成功能 | 等 #5030 合入后，基于最终上游实现 rebase/重新集成，只提交剩余必要适配并复测 |
| 合入前确有协同评审需求 | 可建立显式依赖 #5030 的 stacked draft；注明来源、基线和依赖，避免把同事改动 squash 成自己的迁移工作 |
| 完整验证报告/设计草稿 | 留在 `docs/ascend-a3-validation-20261114`，不把它 merge 到实现 PR |

dlinfer 的 patch 删除与 LMDeploy 原生实现配套：可以关联两个 PR 进行审查，合入/发布清理版时须保证对应 LMDeploy 原生路径已可用，避免旧 LMDeploy 用户先失去 Ascend patch。两个库不要求同步版本号，但应明确最低配套版本与迁移顺序。

提交前查看 `upstream/main...dev/ascend-a3-native` 的实际 diff；上游更新时先处理冲突，再跑受影响的验收项。此次没有创建或更新任何上游 PR，也没有把当前上游 HEAD 冒充实测基线。

## Monokernel 的分支选择

**基于现在的原生迁移代码，另开研发分支，例如 `experiment/ascend-a3-decode-megakernel`。** 可以立刻开发，不必等待迁移 PR 合入；迁移合入后再 rebase 到 main。不要把算法/执行方式优化混进这次保持语义的迁移提交。

同一仓库的新分支能复用现有模型加载、缓存布局、调度入口、数值 harness 和回退路径。初期不需要另起推理框架或恢复 dlinfer patch；Mooncake 是可选集成项，不作为首版 decode megakernel 的必要依赖。

这里把 monokernel 目标理解为一次 decode step 内，用一个或少量较大的 kernel 承担一段模型计算；是否进一步把多层或多 token 驻留在一个 persistent kernel 中，应由测量和平台可实现性决定。

## 建议的首版边界

1. **先测当前 NPUGraph decode。** 分开测 engine 固定批次、NPU device time 和 HTTP streaming，确定时间花在 launch、同步、HBM、矩阵运算还是通信。不能从现有 HTTP C4 观测直接推出需要 megakernel。
2. **先固定模型与形状。** 从已验证的 Qwen3-8B dense、TP1、BF16、固定 batch bucket、MTP 关闭开始；在相同请求/缓存/采样下与当前原生 graph 路径比较。Sage/Agents 的 GDN、MoE 和多卡通信后续加入。
3. **先实现少量融合段。** 候选边界包括 norm/projection/RoPE/KV 写入、attention/output/residual、MLP；它们是待 profiling 选取的候选，不能预先保证全部融合有利。保留通信、动态调度和复杂状态处理的明确边界。
4. **使用现有平台扩展入口。** 实现归入 `kernels/ascend`，包装/选择归入 `backends/ascend`；组件级融合通过现有 BuildSpec 选择，更大的 decode 执行单元在现有 GraphRunner/模型显式入口接入。不要强行把整个模型塞成一个普通算子，也不另外搭通用 Kernel 层。
5. **保留回退与状态语义。** 首版显式 opt-in；不支持的 dtype、batch、context、TP 或模型走当前实现。KV/GDN 地址布局、prefix 状态和请求隔离保持现有契约。MTP 接受/拒绝及状态回滚单独接入和验证。

Ascend C 文档区分 Cube/Vector 计算及其片上存储、核间同步机制。因此本机 A3/CANN 9.1 的可用同步 API、存储预算、驻留/退出行为应先做原型验证；不能直接按 CUDA persistent CTA 的假设设计。这里是设计推断，不是本机 megakernel 能力测试结论。[Ascend C 编程模型](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cpp_tensor_programming_overview.html)

可借鉴 MPK 的任务依赖与 persistent runtime 思路，但该论文实现与测量基于 GPU/CUDA，不代表 A3 上可复用或有相同收益。[MPK 原论文](https://arxiv.org/abs/2512.22219)

## 另行制定的验收

- 正确性先覆盖固定输入/历史的输出与 logits、连续 decode 的 KV 写入、请求切换和边界 shape；既检查数值误差，也检查错误、挂起和状态串扰。MTP 初版关闭，不用其已有非重复性解释新的非 MTP 差异。
- 性能对照当前 NPUGraph，固定相同模型、dtype、batch/context、设备和 warmup；同时报告 decode step 延迟、总 token/s、内存、compile/capture 成本，HTTP 单独衡量。
- 固定一组性能/数值验收阈值后再测量，涵盖目标小 batch 与回退范围；不能只报告单个有利 shape。
- 在功能和收益稳定后，再扩展 TP、GDN/MoE、MTP 和 Mooncake/状态恢复；每一步保留可单独比较的 commit。

当前建议是“独立分支、沿用原生框架、先少量 megakernel 验证收益，再评估更大范围融合”。单个巨型 kernel 是待验证的实现选项，不是预先承诺的完成标准。
