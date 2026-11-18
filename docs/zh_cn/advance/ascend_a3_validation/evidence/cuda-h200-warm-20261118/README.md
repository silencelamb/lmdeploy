# H200 本轮复现依据

这是固定实验快照，不追踪最新 main。公开父版本为 `442097e83799c2e85df9d6bbce35c7c79ea05dfe` 与 `f64ab5061153b5c5b63721cf2d8d6f1b41b95104`；GLM 保序控制为 `f06316de64fe8d3c23f72d17bd94180965a4558e`、`5e51b6718ec31dbcbdbc5fc8092a71b3294e710d`。后两个提交仅在各自父版本增加同一个 selector 参数。

## 恢复证据

按 [archive-manifest.json](archive-manifest.json) 的顺序拼接 `.partNNN`，核对各部分及完整归档 SHA256 后再解包。归档内 `files.json` 列出数据文件长度与 SHA256；索引自身另计一个成员。原始证据目录只读保存，重跑使用新的实验目录和标签。[归档复核](archive-verification.json)记录逐文件验证结果。

可直接查看[短工作负载统计](performance-summary.json)、[32K控制](glm-long-performance-summary.json)、[保序开关成本](selector-cost-summary.json)、[逐批计时](timing-samples.csv)和[最终覆盖/资源核验](final-verification.json)。完整日志与逐token/score记录在raw归档中；权重不包含在归档内。

缓存包保存实际引用的编译产物与原始索引。迁移到其他绝对目录时，在新副本上运行：

```bash
python3 relocate_triton_cache.py --cache /path/to/triton-cache --manifest /path/to/triton-cache-manifest.json
```

工具先验证原始哈希，再仅修改 group 索引的绝对路径，不更换 kernel 或调优 winner。本轮计时期间没有执行重定位。

## 源码与环境

先取得两个公开父提交，再导入随附的精确研究提交：

```bash
git bundle verify /path/to/glm-stable-source-controls.bundle
git fetch /path/to/glm-stable-source-controls.bundle 'refs/validation-export/*:refs/validation-export/*'
```

在独立实验根目录按 `base`、`pr-f64`、`glm-det-base`、`glm-det-pr` 创建四个工作树。使用保存的 CUDA 依赖副本或相同 wheel 集合，并核对 `dependency-versions-full.json`。不要以重新解析移动中的 requirements 替代固定环境。每个进程以显式 `PYTHONPATH` 导入所选工作树，实际路径与源码 SHA 在命令和 import 审计中。

checkpoint 路径、快照、文件 SHA 和本轮元数据见 inventory 与 `checkpoint-identity.json`；权重不在归档中。大模型完整 TP8；Sage、Agents、Decision 使用记录的单卡。设备、dtype、cache blocks、prompt、输入/输出长度与采样条件须匹配。

本轮使用 CUDA profile，不安装或导入 dlinfer、torch_npu、triton-ascend，不执行 Ascend requirements 或 A3 启动脚本；进程导入守卫保留这些约束。

## 执行与核对

- 实际命令、环境覆盖、脚本哈希在 `logs/<label>.command.json`，对应版本保存在 `evidence/script-snapshots`。`scripts/` 另提供最终复现及审计工具。
- 使用新实验根目录或新的 label/output。已有完整 label 会复用旧记录，这不算一次新测量。
- `reperf_queue.py` 执行生成模型 pipeline/HTTP，`reperf_decision.py` 执行真实评分，`reperf_norm.py` 执行算子对照。调度见 `reperf_campaign.py` 与保留历史的 `reperf_resume.py`。
- `prerequisites/` 是原实验已有的数值门槛证据，不替代新机器验证。本轮每个模型进程仍实际采集计时外的完整 token/top-k 分数，并核对计时生成历史。
- 完成引擎预热后，还需实请求预热；HTTP 要覆盖完整 C4 prefill/tuning 形状。仅 `/health` 成功不足以证明稳态。分数捕获只在计时外启用，计时前恢复原始 `AsyncEngine.generate`，并保存控制时间线。
- 每格五对全新进程 AB/BA；每进程每并发五个计时批次，先按进程汇总再配对统计。TTFT/TPOT 使用批内请求均值。
- `selections/` 仅按外部 GPU 进程重叠排除样本，并按相同顺序重跑整对；没有按速度或数值结果挑选样本。
- 主、长上下文、选项开销、资源分析分别由 `reperf_analyze.py`、`reperf_analyze_glm_long.py`、`reperf_analyze_selector_cost.py`、`reperf_audit_resources.py` 完成；`reperf_final_verify.py` 核对覆盖、源码、checkpoint、缓存与选定进程。

保序开/关开销与迁移对照分开：固定输入单输出 token 用于完整模型 prefill 成本；32-token 生成只有实际历史一致时才解释端到端成本。默认 selector 的不重复分数不作为稳定选项的数值 oracle。

通过范围以最新矩阵为准。旧百分比、受干扰尝试和原始失败保留为历史证据，不替代本轮结论。
