# 同事 Ascend PR 接入与 rebase 策略

2026-11-14，Asia/Shanghai。已查询 wanfengcxz 在 InternLM/lmdeploy 与 DeepLink-org/dlinfer 的未关闭 PR，读取以下五个 Ascend PR 的元数据、文件清单及关键 diff。除 Mooncake 的既有实测外，其余条目是源码审查和接入计划，没有宣称已移植或通过硬件验证。

## “原生后端”的含义

本项目中的原生后端就是**已经迁移到 LMDeploy 内部的 Ascend PyTorch 后端**：平台包装在 `lmdeploy/pytorch/backends/ascend`，必要算子和辅助代码在 `lmdeploy/pytorch/kernels/ascend`，通过现有 OpsBackend / BuildSpec / GraphRunner 与显式模型入口调用。运行这个路径不安装或导入 dlinfer；torch_npu、CANN、Triton Ascend 和已记录的外部 kernel 依赖仍保留。

两个 fork 的迁移分支现在均为 `dev/ascend-a3-native`。原始 commit 和验收数据保持不变；改名之后仅更新当前文档引用。

## 当前 PR 清单

| PR | 查询状态 / 固定 head | 处理建议与边界 |
| --- | --- | --- |
| [LMDeploy #5030：Mooncake](https://github.com/InternLM/lmdeploy/pull/5030) | open draft；`1a09270b0835bb871be4eb4df70f9d04c8ccaec2` | 已在独立集成分支按此 head 适配和实测；继续跟踪最终上游版本，不重复搬运 |
| [LMDeploy #5001：GLM-5.2 W8A8 优化](https://github.com/InternLM/lmdeploy/pull/5001) | open；`8e559f484a54adfd6e2b97e3c1a7a268f474a62f` | 与 dlinfer #355 作为一组移植；旧 backend/kernel 包装改接原生 BuildSpec 与 Ascend 实现 |
| [dlinfer #355：MC2 MoE / MLAPO / NZ](https://github.com/DeepLink-org/dlinfer/pull/355) | open；`9102cc598a7af2b88652b7930ddbf22740c70a9b` | 独立算子与 LMDeploy 模型适配分别归位；不能恢复已经移除的 import-time framework patch |
| [dlinfer #324：gated RMSNorm 优化](https://github.com/DeepLink-org/dlinfer/pull/324) | open；`f6ac78b38818d0f2fc16302b19ad3e3e571a27fe` | 原生算子有对应位置；先核对语义/适用条件，再独立验证数值和性能 |
| [dlinfer #295：AWQ 修复](https://github.com/DeepLink-org/dlinfer/pull/295) | open；`1974cec23a3839d032d4befe172c88bd1d557151` | 当前共享 AWQ 已有 TP 初始化和 layer_type 传递；dtype 接口还需按现有调用链核对，只移植实际缺失部分 |

同次查询也返回了 #5093（tool parser）和 #5123（count_tokens），它们不属于本次 Ascend backend 迁移清单。元数据/文件列表固定在 [快照](evidence/incoming-ascend-prs-20261114.json)，后续 PR 更新需重新计算差异，不能沿用本表 head 当成最新版本。

## 源码审查得到的具体注意点

### GLM #5001 / #355 是一个功能组

#5001 增加融合 W8A8 MoE、router、split sparse attention，并修改 A3 prefill 的 MC2 选择及容量计算。后者改变执行策略，属于新增优化，不能混入本次只保持计算语义的迁移基线。

#355 增加 `vendor/ascend/mlapo.py`、融合 MoE 和模型 monkey patch。迁移时建议分为：

- 算子实现、scale 打包、NZ layout 辅助放入 `kernels/ascend`；MoE/router/attention 的选择和包装放入 `backends/ascend`，沿用 BuildSpec。
- 合并 QKV 投影、加载后权重准备、MLAPO fast path 和 NZ 权重转换使用模型或 backend 的显式入口。保留原逻辑的适用条件与 fallback，避免再次给模型类动态覆盖方法。
- dlinfer 若继续提供独立 vendor API，可以保留其独立算子更新；新 LMDeploy 路径必须直接使用已迁入的实现，不能通过安装 dlinfer 获得这些新增能力。

关键新增依赖：读取的 #355 源码会查找并加载 `vllm_ascend_C*.so`，调用 `_C_ascend.dispatch_ffn_combine` 和 `_C_ascend.mla_preprocess`。因此“搬 Python 代码”还不够，必须确认这些二进制算子的来源、版本、CANN/torch_npu ABI、安装方式及缺失时的回退。没有导入 vLLM Python 不等于没有外部二进制依赖。

#355 还把 dlinfer 的 Python 版本上界改到 `<3.14`；这项 packaging 改动应单独审查，不能用它把现有镜像的全部依赖约束问题自动判为解决。

PR 作者报告过 16 卡 TP16/EP16 的 GLM 测试；这是作者在其环境的验证。本机可见 8 个 NPU chip，不能据此替代 16 卡场景的验证，也不能把当前 Sage/Agents BF16 结果外推到 GLM W8A8。

### #324 需要保留 RMSNormGated 的接口语义

该 diff 将 `rmsnorm_fn(...)` 改成 NPU RMSNorm 后乘 FP32 SiLU，再转回输入 dtype。新实现没有沿用 `group_size`、`norm_before_gate` 的分派，并直接使用 `z`。当前原生 backend 对已用 GDN 路径显式选择 norm-before-gate，但保留的 RMSNormGated 类仍有更广的参数接口。

因此移植前应检查分组归一化、gate/norm 顺序、`z=None` 和 dtype/舍入行为；可把优化限制在已证明等价的组合，其余回退。PR 展示的 batch 128 性能不能直接代表本机小 batch decode 或迁移收益。

### #295 先识别已覆盖的内容

该 diff 为旧的 `AscendMergedAwqLinear__init__` 补充 `layer_type`、`dtype` 和 TP 初始化。迁移后的共享 `MergedAwqLinear` 已有 `init_tp_args(..., layer_type=...)` 以及父构造传递；当前 `AwqLinear` 仍显式使用 FP16，Merged 构造没有独立 dtype 参数。

所以这不是把旧 `ascend_awq.py` 放回 dlinfer 的理由。应检查实际 caller 与 checkpoint 的 dtype 要求，在当前 `nn/linear/awq.py` / Ascend backend 的明确入口补需要的差异，并复核 TP 分片、QKV/merged 权重加载与 CUDA 公共路径。本轮只做源码审查，尚未完成 AWQ 模型验证。

## Rebase 与功能移植各做什么

**Rebase 同步上游基线；功能移植负责让新后端真正执行同事的新增能力。** Git 合并没有冲突，也不代表新代码已接入 Ascend 的实际执行路径。

建议由你统一负责原生适配，同时保持每组功能的来源和独立提交：

1. `dev/ascend-a3-native` 保留迁移与必要修复，作为当前功能对照基础。
2. 从它分出各功能分支。建议名字为 `dev/ascend-a3-awq-followup`、`dev/ascend-a3-gated-rmsnorm`、`dev/ascend-a3-glm52-w8a8`；这些只是命名建议，本轮没有创建或实现它们。
3. 逐项固定原 PR head，先检查 main/当前 native 是否已有等价修改；可直接复用的提交保留作者与原 SHA 引用（例如 cherry-pick -x）。路径/架构变化较大的内容做明确的 port commit，在正文链接原 PR/commit、记录保留和调整的语义，不把原作者的工作混成无来源的大提交。
4. 针对每组功能，对比“旧框架 + 同事功能”和“原生后端 + 同一功能”；优化收益另与未启用优化的 native 比较。新旧都必须启用同一功能，才能区分功能收益与迁移效果。
5. 每组功能独立验收、独立 PR。#5001/#355 保持配套，Mooncake 保持已有集成分支，monokernel 保持单独研发分支。

若**同事的功能先合入 upstream/main**：将你的迁移分支 rebase 到新的 main，再补齐那些落在旧 dlinfer 路径、没有自动进入原生路径的功能，重新跑受影响对照。

若**你的迁移先合入 upstream/main**：后续功能 PR 基于新的 main，直接适配到原生 backend。此时可以由你帮助他完成 port 和回归，不必持续维护一条只给旧框架用的实现。

尚未合入的功能可在 fork 上提前以依赖迁移的 feature branch 开发/评审。建议先核对小范围 AWQ 缺口，再做 gated RMSNorm，最后处理成对 GLM 优化；实际顺序可按业务优先级调整。无需为了这些后续优化反复把整个迁移 PR rebase 到每个同事的功能分支。
