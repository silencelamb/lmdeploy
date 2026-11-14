# Ascend A3 迁移验证与提交记录

更新：2026-11-14，Asia/Shanghai。原始运行时间保留 UTC，便于与日志对应。

本目录仅在用户 fork 的 `docs/ascend-a3-validation-20261114` 分支维护。上游迁移 PR 使用 `codex/ascend-a3-native`，不要把整个文档分支合入 PR。

## 当前结论

已完成代表性 A3 迁移对照，修复一个明确的 kernel 导入启动回归。Sage-A1.5、Agents-A1 的已测非 MTP 数值、prefix cache 与近 256K 上下文检索通过；原生环境没有安装或导入 dlinfer。独立 dlinfer 的算子验证另行执行。

| 项目 | 实测结论 | 当前处理 |
| --- | --- | --- |
| 原生 Ascend 启动 | 延迟导出的 callable 被同名子模块覆盖；`6421cefe` 修复，CPU contracts 33/33 通过 | 已修复 |
| 非 MTP 数值与缓存 | Sage/Agents 的已测 token/logprob 一致；真实 prefix 命中；两个模型 261845-token 检索通过 | 已测场景通过 |
| Sage MTP 长生成 | MTP3 graph 新旧 13/21 序列一致；独立固定种子重复中旧基线自身也仅 14/21 一致 | 严格等价未通过；归因未完成；维护者接受为非阻塞观察 |
| HTTP C4 | 总吞吐超过 5% 的下降未在反向复测重复确认；流式输出速度两轮均下降，Sage 12.18%/5.87%，Agents 7.13%/8.99% | 未定位迁移新增原因；维护者接受为非阻塞观察 |
| Agents 固定 C4 引擎对照 | 同请求、同输出长度的 pipeline 对照没有复现下降，28/28 token 序列一致 | 不把 HTTP 指标直接外推为底层计算回归 |
| Agents C1 TTFT p95 | 两轮 +5.96%/+5.07%，约 39/35 ms，小样本尾部分位数 | 保留观察，不宣称已修复 |
| Mooncake 0.3.13.post1 | 独立分支使用 PR #5030 适配；fabric 真正写读、跨实例 KV 命中/GDN 状态恢复通过；51/51 对应响应与 2040 个分数完全一致 | 此路径通过；旧“宿主驱动阻塞”判断已被取代 |
| TileLang 补齐 | 两环境均安装 0.1.15 及固定依赖，worker 套件各 101/101 通过 | CPU 单测缺依赖已解决；补依赖后未重跑整模型/HTTP |
| dlinfer 独立功能 | 无 LMDeploy；16 个 CPU contracts、6 个 >1024 专家 MoE eager/真实 NPUGraph case 通过 | 已测场景通过 |
| 共有失败与未覆盖 | 三项 MLA keyRope 连续性失败、MTP 输入评分失败、镜像依赖约束；CUDA、多机、量化等未覆盖 | 保留失败/未验证，不计为通过 |

**维护者决策与测试事实分开记录。** 2026-11-14，维护者判断 C4 性能现象和基线已有的 MTP 不一致不阻塞本次迁移整理与提交。本记录据此调整提交决策，没有改变原始阈值、JSON 结果或失败计数；“非阻塞”不等于已证明所有配置无回归。

## 分支与提交

| 仓库 / 分支 | 发布源码 HEAD | 用途 |
| --- | --- | --- |
| [LMDeploy `codex/ascend-a3-native`](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-native) | [`c5a7e1b5`](https://github.com/silencelamb/lmdeploy/commit/c5a7e1b532c9be1b849478b09e816495b236a090) | 原迁移 + 已测导入修复 + 简短验证说明；不含 PR #5030 |
| [dlinfer `codex/ascend-a3-native`](https://github.com/silencelamb/dlinfer/tree/codex/ascend-a3-native) | [`7e84b71`](https://github.com/silencelamb/dlinfer/commit/7e84b71c8f83cbd41c919701b3920a7962759f30) | patch 清理 + A3 独立功能说明 |
| [LMDeploy `codex/ascend-a3-native-mooncake`](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-native-mooncake) | [`b9ee85de`](https://github.com/silencelamb/lmdeploy/commit/b9ee85debecda131e2e4b0b6f6048aa8029b8855) | 原生迁移 + 同事 PR #5030 的独立集成验证 |
| [LMDeploy `codex/ascend-a3-pr5030-baseline-20261113`](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-pr5030-baseline-20261113) | [`b334d7aa`](https://github.com/silencelamb/lmdeploy/commit/b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d) | 旧基线加同一 Mooncake 适配，便于公平复现 |
| LMDeploy `docs/ascend-a3-validation-20261114` | 当前文档分支 | 完整报告、结果、命令、设计建议；不进入迁移 PR |

发布中的简短说明/merge 没有改变实测计算代码。实际比较 commit 必须固定如下，不能用未来移动的 branch HEAD 替代：

| 对象 | 实测 commit |
| --- | --- |
| LMDeploy 旧基线 | `fb5de87a45df9999a19a569ed9441df8edfa6975` |
| dlinfer 旧基线 | `3b9096341a41e9d067104a39d914e47ca2d7ec50` |
| LMDeploy 原迁移 | `39dbaa841080d9b9b6b0326197b9de7fe9495a1b` |
| LMDeploy 导入修复后 | `6421cefea94edd350e419f556158639556357a27` |
| dlinfer 清理后 | `05ede38ae879347be33c8f6ebb136d84f3f4ee88` |
| Mooncake 对照基线 | `b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d` |
| Mooncake 原生集成 | `4930080caac39c73427fcfd083d082fdebf7ec09` |
| 同事 Mooncake PR head | `1a09270b0835bb871be4eb4df70f9d04c8ccaec2` |

## 报告与复现

- [第一阶段：迁移、精度、性能、dlinfer 独立验证](migration_validation.md)。保留旧结论上下文；其中 Mooncake 阻塞判断以第二阶段为准。
- [第二阶段：Mooncake fabric 与 TileLang 补测](mooncake_validation.md)。实测 PR 适配、外部缓存数值、C1 延迟及 101/101 单测。
- [复现入口与证据说明](reproduce.md)。包含固定版本、模型/环境前提及执行命令。
- [PR 顺序与 monokernel 设计建议](next_steps.md)。这是建议和设计边界，没有实现或性能收益声明。
- [逐文件来源/哈希](evidence/source-index.json)、[发布源码版本](evidence/published-code-revisions.json)、[PR 状态快照](evidence/pr5030-status-20261114.json)。

Git 中保存选定的数值摘要、CSV、JUnit XML、运行命令、环境/模型清单和 Markdown 形式的原始 harness 快照。完整原始日志、请求 token/logprob、wheel 与大归档仍在 A3 本机；[归档路径、大小与 SHA256](evidence/local-archives.json) 明确标注未上传。原始部署文档及同事目录没有整体复制到公开仓库。

## 维护方式

新增测试使用新的日期/运行 label，保存完整配置和原始数据，再更新本总览；不覆盖旧失败记录。新的集成依赖或性能优化分别使用独立分支和结果集。将来提 PR 只选择迁移代码分支；若需要给上游补简短验收说明，单独摘取，不 merge 本文档分支。
