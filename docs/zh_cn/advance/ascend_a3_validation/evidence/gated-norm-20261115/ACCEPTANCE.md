# Gated RMSNorm 重构验收

固定原生重构前版本：`342db85ecff3c0dc864a4bf8613c589c270f4141`。不修改历史验证环境、模型文件或同事目录。

- 公共层拥有 weight，backend 实现不拥有 Parameter；checkpoint 的 norm.weight/o_norm.weight 名称及直接缩放语义不变。
- 三个直接调用方和 Qwen3.5 派生模型全部使用公共层。覆盖 Qwen3-Next、Qwen3.5 dense/MoE、InternDecision、Intern-S2 Preview/Mobius、MetaMoe、GLM5-Next；核对 MTP 子模型是否实际包含此算子。
- CUDA 调用与原 FLA 0.5.2 模块相同的 functional 入口，保留 NB autotuning workaround、silu/swish/sigmoid、affine 和 residual/prenorm 参数。CUDA 实机测试只有实际运行才能记为通过。
- Ascend 已有 SiLU、BF16 路径与冻结的旧 kernel/旧模块逐位比较。sigmoid 是此前工厂忽略 activation 的缺口，使用独立 FP32 数学公式验收，并明确作为修复单列。
- NPU 数学参考容差预设：BF16 rtol=0.02/atol=0.02，FP16 rtol=0.003/atol=0.003，FP32 rtol=0.00001/atol=0.00001。旧、新实现同参数比较要求字节相同，不用容差。
- 验证形状、非连续输入、dtype、权重加载、NPUGraph，以及旧 kernel 的分组、归一化顺序与无 gate 能力。
- 对本机可运行的代表性 checkpoint 做同环境重构前后对照，非 MTP token 和完整已记录 top-k 分数要求逐位一致；MTP 的基线非确定性单列。不能以算子或小模型测试替代所有完整 checkpoint 认证。
- 新路径禁止导入 dlinfer。检查公共 CPU 契约及硬件测试选择，执行适用 lint 与打包/导入检查。

- 增量性能观察：Sage/Agents 同配置 warmed C1/C4，各 3 次；吞吐中位数下降超过 5% 或 TTFT 中位数上升超过 10% 时追加确认。短样本仅用于发现重构回归，不替代部署容量认证。
