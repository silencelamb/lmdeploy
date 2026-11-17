# Ascend A3 迁移验证与提交记录

PR GPU结果汇总与独立说明：[三类CUDA基线一致性问题、六模型结论和S2归约提交来源](cuda_consistency_findings_20261117.md)。

进一步CUDA机制分析：[GLM原子索引、3+1动态合批与S2 BF16 Ring归约](cuda_determinism_followup_20261117.md)。新增冻结算子证据，不替代完整模型修复验收。

待执行的独立任务：[GLM-5.3保序与S2确定性TP归约修复Prompt](cuda_determinism_independent_task_prompt_20261117.md)。基线问题另开实现分支，单机8卡验收，不混入迁移PR。

H200 CUDA固定版本对照已收尾：[完整报告与原始证据](cuda_regression_20261116.md)。固定442097/f64；匹配调优缓存下五个完整checkpoint的迁移对照通过，GLM-5.3长上下文为基线问题；HTTP动态合批/排列、prefix跨命中状态等价性和插桩退出限制单列。该结果不等于所有配置全通过。

更新：2026-11-16，Asia/Shanghai。原始运行时间保留 UTC，便于与日志对应。

本目录仅在用户 fork 的 `docs/ascend-a3-validation-20261114` 分支维护。上游迁移 PR 使用独立的 `pr/ascend-a3-native`，开发分支保留为 `dev/ascend-a3-native`。不要把整个文档分支合入 PR。

原生迁移已提交：[InternLM/lmdeploy #5134](https://github.com/InternLM/lmdeploy/pull/5134)。范围与本轮检查见 [PR 整理记录](native_pr_20261115.md)，完整对比报告通过链接提供，未加入 PR。

最新CI恢复：PR HEAD为`2ccf38a4`，**14/14项门禁全部通过**（13项Actions及Read the Docs）。Jetson的cuDSS版本和库搜索路径问题已修复；8个CUDA组合、A3、lint、完整单测和ETE均完成。单测7452通过/400跳过，ETE按阶段合计351次通过。[CI修复与完整证据](ci_docker_recovery_20261116.md)保留失败迭代、RTD重建和验证范围。CANN继续保持9.1.0。

本次补充：[RMSNormGated 标准接口重构、历史来源与完整调用者验证](gated_norm_refactor_20261115.md)。公共层现在管理权重，后端只计算；明确补齐 sigmoid/dtype，保留旧 SiLU 数值路径。

最新复查：[Agents gated norm 性能定位](gated_norm_performance_20261116.md)。实际旧/新 NPU kernel 二进制相同，交替对照未稳定复现原来的 C4 −5.15%；流式长停顿已在主机输入提交等待处复现，完整证据与尚未解决的限制单列。

最新交付：[MTP编译器根因、升级验证与PR集成](mtp_ray_investigation_20261116.md)。状态缓冲与同步事件轮转层级错配已定位，官方NPUIR2.0.0验证通过；circular GDN单缓冲修复已纳入PR #5134的`85a05812`。最终PR源码在旧/新编译器下各6项NPU回归、各两轮21响应的完整MTP重复性通过。CPU18的具体softirq/tasklet回调仍待有权限的宿主机取证，亲和性规避不是永久修复。[Ray后续计划](mtp_ray_investigation_20261116.md#ray-后续调查计划)明确了实验顺序与完成判据；[H200报告](cuda_regression_20261116.md#ray-独立-cpu-对照)新增两轮纯Ray CPU对照，有限窗口未复现，不能据此归因ARM。固定版本CUDA结果见上述H200报告。

2026-11-14 两个 fork 的迁移主分支已由 `codex/ascend-a3-native` 正式改名为 `dev/ascend-a3-native`；提交历史保留，当前说明和链接已更新。原始证据中的旧分支名是当时的记录，不作回写。Mooncake 集成分支和报告分支沿用原名。

## 当前结论

已完成代表性 A3 迁移对照，修复一个明确的 kernel 导入启动回归。Sage-A1.5、Agents-A1 在已测相同非 MTP 配置下的生成 token ID、完整 top-k token 集合与已记录 logprob 分数按位一致；prefix cache 与近 256K 上下文检索通过。原生环境没有安装或导入 dlinfer，独立 dlinfer 的算子验证另行执行。

| 项目                    | 实测结论                                                                                                                                       | 当前处理                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| RMSNormGated 标准接口   | 98 项 CPU、55 项 A3、32 项旧实现逐位检查、208 层真实权重通过；五模型记录输出逐位一致；A100 CI 新增 norm 文件 83/83 通过（含 56 项 CUDA/graph） | 详见 [重构报告](gated_norm_refactor_20261115.md)及[CI 补充](gated_norm_performance_20261116.md#pr-与-ci-记录)，固定版本GPU模型对照见H200报告 |
| 重构后性能观察          | 原 C4 −5.15% 未稳定复现；实际 kernel 二进制相同；同 worker 8 对交替测量 C1/C4 吞吐中位数变化 −1.40%/+10.88%，区间较宽                          | [性能定位](gated_norm_performance_20261116.md)：主机提交等待已复现，未宣称性能全面通过                                               |
| 原生 Ascend 启动        | 延迟导出的 callable 被同名子模块覆盖；`6421cefe` 修复，CPU contracts 33/33 通过                                                                | 已修复                                                                                                                               |
| 非 MTP 数值与缓存       | Sage/Agents 的已记录 token ID、top-k 集合及分数按位一致；真实 prefix 命中；两个模型 261845-token 检索通过                                      | 已测场景通过；原始输出的二进制复核也通过                                                                                             |
| Sage MTP 长生成 | 历史基线也不稳定；现已定位到NPUIR自动多缓冲状态写回错误。PR修复在旧/新编译器各两轮21响应及top5分数逐位一致 | `85a05812`纳入circular单缓冲和硬件回归；已测范围通过，其他MTP配置仍未覆盖 |
| Agents 固定批次引擎对照 | 同请求、同输出长度的 pipeline 对照中，28/28 token 序列一致                                                                                     | 已测场景通过                                                                                                                         |
| Agents C1 TTFT p95      | 两轮 +5.96%/+5.07%，约 39/35 ms，小样本尾部分位数                                                                                              | 保留观察，不宣称已修复                                                                                                               |
| Mooncake 0.3.13.post1   | 独立分支使用 PR #5030 适配；fabric 真正写读、跨实例 KV 命中/GDN 状态恢复通过；51/51 对应响应与 2040 个分数完全一致                             | 此路径通过；旧“宿主驱动阻塞”判断已被取代                                                                                             |
| TileLang 补齐           | 两环境均安装 0.1.15 及固定依赖，worker 套件各 101/101 通过                                                                                     | CPU 单测缺依赖已解决；补依赖后未重跑整模型/HTTP                                                                                      |
| dlinfer 独立功能        | 无 LMDeploy；16 个 CPU contracts、6 个 >1024 专家 MoE eager/真实 NPUGraph case 通过                                                            | 已测场景通过                                                                                                                         |
| 共有失败与未覆盖        | 三项 MLA keyRope 连续性失败、MTP 输入评分失败、镜像依赖约束；CUDA、多机、量化等未覆盖                                                          | 保留失败/未验证，不计为通过                                                                                                          |

2026-11-15 已按维护者要求更新 PR 验证摘要。历史测量、原始阈值与 JSON 结果保持原样归档；当前提交说明与 GPU 后续计划见 [PR 整理记录](native_pr_20261115.md)。GPU 待测模型为 GLM-5.3-Flash、InternDecision、Sage-A1.5、Agents-A1、Intern-S2、GLM-5.3，均要求同一 GPU 环境下迁移前后按位一致，尚未计为通过。

## 分支与提交

| 仓库 / 分支                                                                                                                                  | 发布源码 HEAD                                                                                         | 用途                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [LMDeploy `pr/ascend-a3-native`](https://github.com/silencelamb/lmdeploy/tree/pr/ascend-a3-native)                                           | [`2ccf38a4`](https://github.com/silencelamb/lmdeploy/commit/2ccf38a43f0fdc7415ce4ce9f177852bdacc9966) | 迁移、GDN/CI修复与CANN9.1依赖配套 PR；不含完整对比文档                     |
| [LMDeploy `dev/ascend-a3-native`](https://github.com/silencelamb/lmdeploy/tree/dev/ascend-a3-native)                                         | [`7d984d0a`](https://github.com/silencelamb/lmdeploy/commit/7d984d0abce61849237d436ac4243fc964e93e7b) | 原迁移、已测修复、测试隔离、平台名称与编译策略说明；不含 PR #5030 |
| [dlinfer `dev/ascend-a3-native`](https://github.com/silencelamb/dlinfer/tree/dev/ascend-a3-native)                                           | [`25b81c69`](https://github.com/silencelamb/dlinfer/commit/25b81c69cde38e61afc52c50ea92b1d625793db8)  | patch 清理 + A3 独立功能说明                                      |
| [LMDeploy `codex/ascend-a3-native-mooncake`](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-native-mooncake)                   | [`2c85cdc8`](https://github.com/silencelamb/lmdeploy/commit/2c85cdc888e313c4c32a15e7f7b2a81cf4d95d2b) | 原生迁移 + 同事 PR #5030 的独立集成验证                           |
| [LMDeploy `codex/ascend-a3-pr5030-baseline-20261113`](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-pr5030-baseline-20261113) | [`b334d7aa`](https://github.com/silencelamb/lmdeploy/commit/b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d) | 旧基线加同一 Mooncake 适配，便于公平复现                          |
| LMDeploy `docs/ascend-a3-validation-20261114`                                                                                                | 当前文档分支                                                                                          | 完整报告、结果、命令、设计建议；不进入迁移 PR                     |

2026-11-14 发布中的简短说明/merge 没有改变实测计算代码。11-15 新增硬件测试显式开关与 `ascend` 平台名称收敛，针对性结果见[本轮审查](code_review_20261115.md)，没有重跑完整模型/性能。实际历史比较 commit 必须固定如下，不能用未来移动的 branch HEAD 替代：

| 对象                  | 实测 commit                                |
| --------------------- | ------------------------------------------ |
| LMDeploy 旧基线       | `fb5de87a45df9999a19a569ed9441df8edfa6975` |
| dlinfer 旧基线        | `3b9096341a41e9d067104a39d914e47ca2d7ec50` |
| LMDeploy 原迁移       | `39dbaa841080d9b9b6b0326197b9de7fe9495a1b` |
| LMDeploy 导入修复后   | `6421cefea94edd350e419f556158639556357a27` |
| dlinfer 清理后        | `05ede38ae879347be33c8f6ebb136d84f3f4ee88` |
| Mooncake 对照基线     | `b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d` |
| Mooncake 原生集成     | `4930080caac39c73427fcfd083d082fdebf7ec09` |
| 同事 Mooncake PR head | `1a09270b0835bb871be4eb4df70f9d04c8ccaec2` |

## 报告与复现

- [2026-11-16：Agents gated norm 性能定位](gated_norm_performance_20261116.md)。包括编译二进制核对、同 worker 交替测量、GC 排查、阶段计时、失败实验与覆盖限制。

- [RMSNormGated 重构与全部调用者验证](gated_norm_refactor_20261115.md)。包含来源归属、规范接口、实机检查、真实权重和完整模型覆盖边界。

- [原生迁移 PR 整理与本轮验证](native_pr_20261115.md)。说明 PR 范围、源码来源、118/10/1 扩展结果与上游 CUDA 环境失败对照。

- [2026-11-15：测试来源、CI 隔离与公共接口审查](code_review_20261115.md)。包含 `ascend` 命名修正、无 torch_npu 的 CPU 验证及硬件 opt-in 检查。

- [第一阶段：迁移、精度、性能、dlinfer 独立验证](migration_validation.md)。保留旧结论上下文；其中 Mooncake 阻塞判断以第二阶段为准。

- [第二阶段：Mooncake fabric 与 TileLang 补测](mooncake_validation.md)。实测 PR 适配、外部缓存数值、C1 延迟及 101/101 单测。

- [复现入口与证据说明](reproduce.md)。包含固定版本、模型/环境前提及执行命令。

- [同事 Ascend PR 接入清单与 rebase 策略](colleague_prs.md)。固定五个 PR 的 head，区分已验证集成、源码审查和后续迁移。

- [PR 顺序与 monokernel 设计建议](next_steps.md)。这是建议和设计边界，没有实现或性能收益声明。

- [逐文件来源/哈希](evidence/source-index.json)、[首次发布源码版本](evidence/published-code-revisions.json)、[分支改名及文档更新记录](evidence/branch-rename-20261114.json)、[PR 状态快照](evidence/pr5030-status-20261114.json)。

Git 中保存选定的数值摘要、CSV、JUnit XML、运行命令、环境/模型清单和 Markdown 形式的原始 harness 快照。完整原始日志、请求 token/logprob、wheel 与大归档仍在 A3 本机；[归档路径、大小与 SHA256](evidence/local-archives.json) 明确标注未上传。原始部署文档及同事目录没有整体复制到公开仓库。

## 维护方式

新增测试使用新的日期/运行 label，保存完整配置和原始数据，再更新本总览；不覆盖旧失败记录。新的集成依赖或性能优化分别使用独立分支和结果集。将来提 PR 只选择迁移代码分支；若需要给上游补简短验收说明，单独摘取，不 merge 本文档分支。
