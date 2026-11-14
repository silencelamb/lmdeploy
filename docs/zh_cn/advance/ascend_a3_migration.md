# Ascend A3 原生 PyTorch backend 迁移记录

截至 2026-11-14，已完成下面列出的 A3 代表性对照，NVIDIA GPU 实机回归仍未执行。两个 fork 使用配套分支 `dev/ascend-a3-native`；LMDeploy 承接原生实现，dlinfer 删除已经迁出的 LMDeploy Ascend patch。此分支不包含尚未合入的 Mooncake PR #5030，相关验证使用独立集成分支。

## A3 实测摘要

- 固定旧基线为下表两个 commit；迁移后的实测源码为 `6421cefea94edd350e419f556158639556357a27`。该提交修复 Ascend kernel 延迟导出被同名子模块覆盖的问题，不改变计算体。
- Sage-A1.5 / Agents-A1 的已测非 MTP 数值对照、实际 prefix cache 命中与 261845-token 检索通过；native 环境未安装或导入 dlinfer。原生 CPU contract suite 为 33 passed。
- Sage MTP 长生成在旧基线自身也不严格重复；新旧严格等价验收仍未通过。HTTP C4 总吞吐超过 5% 的下降未在反向复测中重复确认，但逐请求流式速度两轮均出现下降。按维护者 2026-11-14 的决定，作为非阻塞观察项保留；不将这些现象认定为已定位的迁移新增回归，也不改写为已修复。
- 三项 paged/decode MLA 算子在新旧环境均有 CANN keyRope 连续性错误，MTP 输入评分路径也有共同失败。Python 3.12 / torch 2.10 / CANN 9.1.0 镜像超出部分依赖声明，pip check 未通过。未测配置不能外推为通过。

完整报告、固定版本、结果、复现命令及 Mooncake 独立验证维护在 fork 的 [验证文档分支](https://github.com/silencelamb/lmdeploy/tree/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation)，不属于迁移实现的必要改动。下文保留迁移设计、Mac 检查及其他配置的验收步骤；这些步骤不等于所有场景已在 A3 执行。

源码基线：

| checkout | commit |
| --- | --- |
| lmd_fork | `fb5de87a45df9999a19a569ed9441df8edfa6975` |
| dlinfer_fork | `3b9096341a41e9d067104a39d914e47ca2d7ec50` |

迁移清单的“原实现”列引用上述基线 commit；dlinfer 配套清理后，已删除文件应通过 `git show 3b909634:<文件>` 查看。其余位置指迁移后的源码，不把说明文档的兼容性或性能陈述视为实测。

配套分支：[LMDeploy](https://github.com/silencelamb/lmdeploy/tree/dev/ascend-a3-native)、[dlinfer](https://github.com/silencelamb/dlinfer/tree/dev/ascend-a3-native)。dlinfer 清理范围与保留项见其 [迁移说明](https://github.com/silencelamb/dlinfer/blob/dev/ascend-a3-native/docs/ascend_native_migration.md)。

## 迁移清单

沿用现有 `OpsBackend`、`BuildSpec`、`GraphRunner`，没有增加通用算子框架。Ascend 的入口改为本仓库 backend，MACA/CAMB 继续进入旧 dlinfer backend；旧共享目录未删除。入口在 `lmd_fork/lmdeploy/pytorch/backends/selector.py:10`、`lmd_fork/lmdeploy/utils.py:471`。

| 范围 | 原实现 | 新实现 / 明确调用入口 |
| --- | --- | --- |
| 算子选择、attention/MoE/GDN metadata、HCCL、cache 形状 | `lmd_fork/lmdeploy/pytorch/backends/dlinfer/op_backend.py:1`；`lmd_fork/lmdeploy/pytorch/backends/dlinfer/ascend/op_backend.py:159` | `lmd_fork/lmdeploy/pytorch/backends/ascend/op_backend.py:158`，以 DefaultOpsBackend 为基类，保留 Ascend 分支 |
| Attention、RoPE、norm、linear、AWQ、W8A8、MoE、NSA 的 BuildSpec 实现及包装 | `lmd_fork/lmdeploy/pytorch/backends/dlinfer/attention.py:1`、同目录各算子；`lmd_fork/lmdeploy/pytorch/kernels/dlinfer/__init__.py:1` | `lmd_fork/lmdeploy/pytorch/backends/ascend/attention.py:1`、同目录各算子；`lmd_fork/lmdeploy/pytorch/kernels/ascend/__init__.py:6` 延迟导出包装 |
| 公共 op 参数归一化、NPU primitives、dense/MLA/DSA、MoE | `dlinfer_fork/dlinfer/ops/llm.py:1`；`dlinfer_fork/dlinfer/vendor/ascend/torch_npu_ops.py:1`、`attention.py:1`、`moe.py:1` | `lmd_fork/lmdeploy/pytorch/kernels/ascend/ops.py:1`、`npu_ops.py:1`、`attention.py:1`、`moe.py:1`；保留默认参数，直接调用，不依赖 vendor registry |
| 自定义 kernel 与辅助类型 | `dlinfer_fork/dlinfer/vendor/ascend/triton_ops/causal_conv1d.py:1`、同目录 `fla/`、`rms_norm_gated.py`、`reject_sample.py`；`dlinfer_fork/dlinfer/utils/type_annotation.py:1` | `lmd_fork/lmdeploy/pytorch/kernels/ascend/triton_ops/causal_conv1d.py:1` 及对应文件；`types.py:1`。保留原版权和来源注释 |
| 运行时版本、ATB 扩展加载 | `dlinfer_fork/dlinfer/vendor/ascend/version.py:1`；`dlinfer_fork/dlinfer/vendor/ascend/pytorch_patch.py:30` | `lmd_fork/lmdeploy/pytorch/backends/ascend/version.py:23`、`runtime.py:41`；由 backend/init 和原引擎初始化入口显式调用 |
| NPUGraph capture/replay、CPU metadata update、SSM staging | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/cudagraph/ascend_cudagraph.py:1` | `lmd_fork/lmdeploy/pytorch/backends/ascend/graph_runner.py:283`；`lmd_fork/lmdeploy/pytorch/models/utils/cudagraph.py:74` 等四个入口显式分派，保留模型子类扩展 |
| 连续 cache 布局、逐 block 拷贝 workspace | `lmd_fork/lmdeploy/pytorch/backends/dlinfer/cache.py:1`、`ascend/cache.py:1` | `lmd_fork/lmdeploy/pytorch/backends/ascend/cache.py:10`、`:58` |
| Qwen3.5 状态布局、contiguous QKV、gated RMSNorm | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:933`、`:952` | `lmd_fork/lmdeploy/pytorch/backends/ascend/model_config.py:22`；`lmd_fork/lmdeploy/pytorch/models/qwen3_5.py:1132`；native `gated_delta_rule.py:51`；新增 `RMSNormGatedBuildSpec` 在 `lmd_fork/lmdeploy/pytorch/backends/gated_delta_rule.py:292` |
| ModelSlim 描述、W8A8 静态/动态选择、scale/offset 加载 | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:171`、`:329`、`:644` | `lmd_fork/lmdeploy/pytorch/backends/ascend/model_config.py:10`、`modelslim.py:5`；`lmd_fork/lmdeploy/pytorch/config.py:1049`；`lmd_fork/lmdeploy/pytorch/models/deepseek_v2.py:1265` |
| DeepSeek Q/KV-A 分离、GLM norm dtype、BF16 indexer、独立 noPE/RoPE cache | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:289`、`:375`、`:541`、`:763`、`:780`、`:849` | `lmd_fork/lmdeploy/pytorch/models/deepseek_v32.py:252`；`lmd_fork/lmdeploy/pytorch/models/glm_moe_dsa.py:46`、`:215`、`:279`、`:319`、`:375`；`lmd_fork/lmdeploy/pytorch/configurations/glm_moe_dsa.py:11` |
| MoE DP/TP gather/scatter、EP 与共享专家 reduction | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/ascend.py:101`；`dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:574` | `lmd_fork/lmdeploy/pytorch/backends/ascend/moe_communication.py:41`；`lmd_fork/lmdeploy/pytorch/nn/moe/base.py:1` 使用 backend factory；`lmd_fork/lmdeploy/pytorch/models/deepseek_v2.py:796` |
| FP32 sampling、bad words、rejection | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:82`、`:140`；`device/ascend.py:52`；`lmd_fork/lmdeploy/pytorch/backends/dlinfer/ascend/rejection_sampling.py:1` | `lmd_fork/lmdeploy/pytorch/engine/model_agent/agent.py:715`；`lmd_fork/lmdeploy/pytorch/backends/ascend/logits.py:5`、`rejection_sampling.py:51` |
| ChatGLM RoPE、NPU profiler、AWQ 转换与加载 | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/ascend.py:38`、`:382`；`quants/ascend_awq.py:1` | `lmd_fork/lmdeploy/pytorch/backends/ascend/rotary.py:5`、`profiler.py:10`、`quantization.py:14`；`lmd_fork/lmdeploy/pytorch/nn/linear/awq.py:162` |
| InternVL vision helper 编译策略 | `dlinfer_fork/dlinfer/framework/lmdeploy_ext/device/__init__.py:26`、`:68` | `lmd_fork/lmdeploy/pytorch/backends/default/compile.py:10`；两个 InternVL 模型的 decorator 显式选择 CUDA 编译或其他设备 eager |

这次迁移解决的是 A3 推理启动依赖外部框架 patch 的问题；只删安装依赖会在 backend、kernel 导入及模型装配时失效。CUDA 继续使用自己的 backend，其他引擎的对应机制未核实。代价是 LMDeploy 现在维护这份 Ascend 实现，以及外部 NPU ABI、kernel 版本的兼容性。入口和源码位置见上表最后一列。

与直接复制旧 patch 有关的处理：

- 不再赋值覆盖 LMDeploy/Transformers 类或全局 `flash_mla_available`、DSA fusion 配置；平台差异在 builder、组件和 backend factory 内选择。GLM 的 fused-indexer CUDA 分支保留，Ascend 使用原独立 projection/BF16 路径。位置：`lmd_fork/lmdeploy/pytorch/models/glm_moe_dsa.py:72`、`lmd_fork/lmdeploy/pytorch/configurations/deepseek_v32.py:41`。
- GraphRunner 继续调用模型的 graph methods，不能绕过 MTP 子类的 hidden-state buffer；graph key 纳入当前模型的 extra key，区分 GLM MTP seed/reuse 等状态。位置：`lmd_fork/lmdeploy/pytorch/backends/ascend/graph_runner.py:283`、`:397`。
- 旧函数式 rejection patch 已不能接到当前 BuildSpec sampler。默认保留当前 Ascend Torch sampler；迁入的 Triton sampler 通过 `DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON=1` 显式启用。位置：`lmd_fork/lmdeploy/pytorch/backends/ascend/rejection_sampling.py:66`。
- 自定义 kernel 的计算体保留；额外补齐 `fused_recurrent.py` 的 `tldevice` import，原文件的 `FLA_USE_FAST_OPS=1` 分支引用了该名字但未导入。同目录 sigmoid 实现已有相同 import。位置：`dlinfer_fork/dlinfer/vendor/ascend/triton_ops/fla/fused_recurrent.py:11`；`lmd_fork/lmdeploy/pytorch/kernels/ascend/triton_ops/fla/fused_recurrent.py:16`。

## 剩余依赖与边界

| 依赖 / 边界 | 状态与源码依据 |
| --- | --- |
| `torch` / `torch_npu` | 保留，要求 matching major.minor、torch >= 2.8、torch_npu >= 2.8.0.post1 和 `NPUGraph.update`；requirements 上界为 2.10。`lmd_fork/lmdeploy/pytorch/backends/ascend/version.py:23`；`lmd_fork/requirements/runtime_ascend.txt:7` |
| CANN、NNAL/ATB | 保留；`torch_npu/lib/libop_plugin_atb.so` 提供 reshape/cache 等扩展，不能只安装 Python 包。`lmd_fork/lmdeploy/pytorch/backends/ascend/runtime.py:12`；`lmd_fork/lmdeploy/pytorch/kernels/ascend/npu_ops.py:427` |
| `torch_npu.contrib.transfer_to_npu` | 显式初始化时仍加载其 CUDA API compatibility shim；引擎沿用的 stream/device API 需要它。本次去除的是仓库自己的 framework monkey patch，**不声称该 vendor shim 没有进程级副作用**。CUDA 单独进程不会初始化它。`lmd_fork/lmdeploy/pytorch/backends/ascend/runtime.py:41` |
| Triton Ascend | GDN/decode convolution/gated RMSNorm 和可选 Triton rejection 需要它。`lmd_fork/lmdeploy/pytorch/backends/ascend/causal_conv1d.py:15`、`gated_delta_rule.py:54`、`op_backend.py:178` |
| `triton-ascend-kernels` | 原始源码只引用该外部库，两个 fork 内均没有它的 kernel 实现。继续依赖其 chunk GDN 和 L2Norm；不是 dlinfer 依赖。`lmd_fork/lmdeploy/pytorch/kernels/ascend/triton_ops/fla/chunk.py:10`。安装入口为 `requirements/ascend_kernels.txt`；A3 实测固定 `8e70a44afa537fcbcdc6d59e6d2196a38f1e7b6f` 加镜像原有 5 文件补丁，快照与哈希见验证报告。requirements 中的 master 不等于该实测快照 |
| CANN Python prerequisites | 过去由 dlinfer 安装的 requirements 已补入 Ascend runtime requirements；`cloudpickle` 已由 common requirements 提供。`dlinfer_fork/requirements/ascend/torch.txt:5`、`cann.txt:1` → `lmd_fork/requirements/runtime_ascend.txt:11` |
| 旧环境变量 | 保留 `DLINFER_LINEAR_USE_NN_LAYOUT`、`DLINFER_MAX_GROUP_LIST_SIZE`、`DLINFER_MOE_PREFILL_CATCHALL`、`DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON` 的拼写，避免更改既有启动配置；名字不意味着 import 依赖。`lmd_fork/lmdeploy/pytorch/backends/ascend/linear.py:17`、`lmd_fork/lmdeploy/pytorch/kernels/ascend/moe.py:12`、native `rejection_sampling.py:73` |
| 原有不支持项 | 不扩展静态 W8A8 MoE、FP8 indexer 或 fused indexer 支持；仍显式报错或沿用 BF16 fallback。`lmd_fork/lmdeploy/pytorch/backends/ascend/modelslim.py:35`、`nsa.py:164`。独立 DeepSeek-V3.2 的 CUDA fused/Hadamard indexer 不能因为 GLM 适配已迁移就声称可在 A3 完整运行，相关模型端到端能力未核实 |
| 其他平台 | MACA/CAMB 接入和旧共享目录保留；未实机验证这些平台。310P 的旧 patch、A2 Dockerfile 不作为本次 A3 验收对象。`lmd_fork/lmdeploy/pytorch/backends/selector.py:20` |
| 未迁移的旧框架层 | DICP/ATB graph compiler 不在当前 NPUGraph 路径；旧 `torch_npu_ext/aclgraph.py` 为 no-op。独立 Transformers remote-module/DynamicCache 的适配不用于本次重写后的 PyTorch 模型推理，没有重新安装全局 Transformers hook；独立 HF 推理/校准能力不在本次验收范围。`dlinfer_fork/dlinfer/framework/torch_npu_ext/aclgraph.py:2`、`dlinfer_fork/dlinfer/framework/transformers_ext/__init__.py:11` |

## Mac 检查结果

本地环境为 macOS arm64、Python 3.12.12、torch 2.9.0、torchvision 0.24.0、Transformers 5.6.2；没有安装 `torch_npu`、Triton 或 dlinfer。本地 venv 在 `.venv-a3-migration`，已通过 `.git/info/exclude` 排除；测试日志、基线副本和构建产物放在 `.git/` 内。

| 检查 | 实际结果 / 限制 |
| --- | --- |
| Python AST/compile、Ruff、`git diff --check` | 对交付修改执行；不需要 GPU |
| 原生 CPU contract suite | 31 passed，包含导入隔离、BuildSpec 装配、ModelSlim、AWQ 分片、Qwen state、rejection、graph metadata/生命周期/MTP 子类扩展、CUDA helper 编译策略，以及从 dlinfer 旧 allocator 测试接续的混合 dtype 小池回归；没有运行 NPU kernel |
| 扩大公共回归 | 217 passed、51 failed、13 skipped。同样选择在未修改基线上为 186 passed、51 failed、13 skipped；失败 node IDs 和 failure messages 相同，没有新增失败 |
| 51 项失败分类 | 16 项没有 CUDA；1 项没有 FLA；11 项没有 Triton；23 项没有 CUDA CausalConv1d implementation。它们是未完成的硬件/环境相关验证，**不计为通过** |
| A3 hardware suite | Mac 未执行，不把模块级 skip 计为通过 |
| Ascend packaging | sdist 与从 sdist 构建的 wheel 成功；检查 58 个原生源码全部进入两种包；检查 requirements、验证工具进入 sdist；METADATA 无 dlinfer 依赖 |
| CUDA packaging | `DISABLE_TURBOMIND=1` 的 PyTorch-only wheel 可构建，检查依赖与基线一致、没有新增 torch_npu/dlinfer 依赖。不代表编译过 CUDA/TurboMind |
| wheel 导入 | 从隔离安装目录、仓库外工作目录检查 backend 选择；不代表 NPU kernel 可以执行 |
| dlinfer 配套清理 | 16 项 CPU 测试通过；检查 Ascend framework 入口不再安装 patch、其他厂商分派保留、独立 MoE eager/capture 分派。其 NPU runtime 初始化与真实 kernel 未在 Mac 执行 |

运行命令：

```bash
python -m pytest -q tests/pytorch/ascend --ignore=tests/pytorch/ascend/hardware
python -m pytest -q tests/pytorch/ascend tests/pytorch/config \
  tests/pytorch/engine/test_cache_engine tests/pytorch/test_cuda_communicator.py \
  tests/pytorch/spec_decode/test_glm_moe_dsa_mtp.py \
  tests/pytorch/spec_decode/test_cudagraph_strategy.py \
  tests/pytorch/models/test_qwen3_5_norm.py tests/pytorch/nn/test_nsa.py \
  --ignore=tests/pytorch/ascend/hardware
LMDEPLOY_TARGET_DEVICE=ascend python -m build --outdir dist/ascend
LMDEPLOY_TARGET_DEVICE=cuda DISABLE_TURBOMIND=1 python -m build --wheel --outdir dist/cuda
```

## A3 安装与验收

以下命令是迁移时准备的验收模板，在 A3 上、迁移分支的 `lmd_fork` 根目录执行。使用独立 Python 环境，不安装 dlinfer，也不要将其源码目录加入 `PYTHONPATH`。模板沿用原 A3 Dockerfile 的 torch 2.9 / CANN 8.5 组合；实际 A3 对照使用用户提供的 torch 2.10 / CANN 9.1.0 镜像，准确命令与环境见验证报告，不能混用两者的结果。Dockerfile 已改为安装本地构建上下文：`lmd_fork/docker/Dockerfile_ascend_a3:1`。

```bash
# 先按机器实际安装路径加载 CANN 与 NNAL 的 set_env.sh；确认驱动和 A3 可见。
python3.11 -m venv .venv-a3
source .venv-a3/bin/activate
python -m pip install -U pip setuptools wheel build
python -m pip install torch==2.9.0 torch-npu==2.9.0 torchvision==0.24.0
python -m pip install -r requirements/ascend_kernels.txt
LMDEPLOY_TARGET_DEVICE=ascend python -m pip install .
python -m pip install pytest pytest-asyncio
mkdir -p validation/a3
python -m pip freeze > validation/a3/environment.txt
python -m pip check
python tools/ascend/check_runtime.py --require-kernels > validation/a3/runtime.json
python -m pytest -q tests/pytorch/ascend/hardware \
  --junitxml=validation/a3/kernels.xml
```

使用 wheel 时，替换 `pip install .` 为 `pip install /path/to/ascend/lmdeploy-*.whl`，仍需加载 CANN/NNAL，并安装 `ascend_kernels.txt` 的外部库。A3 wheel 和 CUDA PyTorch-only wheel 文件名相同，应放在不同目录并检查 METADATA，不能混用。`check_runtime.py` 会拒绝已安装 dlinfer 的环境并检查 A3 SoC、NPUGraph、ATB、DSA/MC2 和可选 kernel 依赖；实现见 `lmd_fork/tools/ascend/check_runtime.py:10`。

硬件 suite 覆盖原 attention precision、paged MLA graph replay、Lightning Indexer/SFA、split KV 写入、interleaved RoPE，以及 convolution ring reference 测试。ring 测试保留的是 CPU reference 路径，不能用它声称 Triton decode 已通过；后者须由下面的 Qwen pipeline 验证。测试入口见 `lmd_fork/tests/pytorch/ascend/hardware/test_ascend_attention_precision.py:164`、`test_ascend_sparse_attention.py:59`、`test_ascend_causal_conv1d_ring.py:30`。

将模型路径替换成已准备好的同一份 checkpoint，先运行 eager，再分别启动进程验证 graph/prefix：

```bash
python tools/ascend/validate_pipeline.py --model /models/DENSE \
  --eager --output validation/a3/dense-eager.json
python tools/ascend/validate_pipeline.py --model /models/DENSE \
  --output validation/a3/dense-graph.json --compare validation/a3/dense-eager.json
python tools/ascend/validate_pipeline.py --model /models/DENSE --prefix-cache \
  --output validation/a3/dense-prefix.json --compare validation/a3/dense-eager.json

python tools/ascend/validate_pipeline.py --model /models/QWEN35 \
  --eager --output validation/a3/qwen-eager.json
python tools/ascend/validate_pipeline.py --model /models/QWEN35 \
  --output validation/a3/qwen-graph.json --compare validation/a3/qwen-eager.json
python tools/ascend/validate_pipeline.py --model /models/QWEN35 \
  --spec-method qwen3_5_mtp --spec-tokens 3 \
  --output validation/a3/qwen-mtp.json --compare validation/a3/qwen-eager.json
DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON=1 python tools/ascend/validate_pipeline.py \
  --model /models/QWEN35 --spec-method qwen3_5_mtp --spec-tokens 3 \
  --output validation/a3/qwen-mtp-triton.json --compare validation/a3/qwen-eager.json
```

脚本包含长 prompt（跨默认 256-token prefill chunk）、变化的 batch 形状、重复的新请求，保存 token IDs、logprobs 和 cached_tokens，并检查重复请求及参照 trace 的 greedy tokens。`--sampling mixed` 会混合 greedy/random 请求，`--sampling random` 可检查全随机路径能否运行；两者不做逐 token reference 比较，不能代替 rejection 的分布正确性检查。若 greedy token 不一致，保留 trace 并定位首个 logits 偏差，不能放宽断言后宣称通过。实现见 `lmd_fork/tools/ascend/validate_pipeline.py:14`。

```bash
# 同时覆盖采样/logprobs 与 bad-words；再用 Triton 环境变量重复此项。
python tools/ascend/validate_pipeline.py --model /models/QWEN35 \
  --spec-method qwen3_5_mtp --sampling mixed --bad-word forbidden \
  --output validation/a3/qwen-mixed.json
# 原 int8 KV 路径需要真实量化记录，先验证 eager；不能凭空生成 scales。
ASCEND_QUANT_RECORD_FILE=/absolute/path/to/quant_record.txt \
  python tools/ascend/validate_pipeline.py --model /models/DENSE --eager \
  --quant-policy 8 --output validation/a3/dense-int8-kv.json
# 原参数名保留 CUDA，但 Ascend profiler adapter 选择的是 NPU activity。
LMDEPLOY_PROFILE_CPU=1 LMDEPLOY_PROFILE_CUDA=1 \
  LMDEPLOY_PROFILE_OUT_PREFIX=validation/a3/profile_ \
  python tools/ascend/validate_pipeline.py --model /models/DENSE \
  --output validation/a3/dense-profile.json
```

量化记录的读取条件见 `lmd_fork/lmdeploy/pytorch/backends/ascend/op_backend.py:590`；profiler 的 NPU activity 选择见 `lmd_fork/lmdeploy/pytorch/backends/ascend/profiler.py:10`。检查实际导出的 profiler trace，不能只看进程退出码。

| 必须覆盖的 A3 项目 | 执行方式与判据 |
| --- | --- |
| Dense MHA/GQA、DeepSeek-V2 MLA | 对各 checkpoint 运行上述 eager/graph/prefix 三组；attention reference tests 无失败 |
| GLM DSA BF16、ModelSlim W8A8 dynamic/static linear | 对各 checkpoint 运行 eager/graph；使用 GLM checkpoint 的 MTP 权重时加 `--spec-method deepseek_mtp --spec-tokens 3`；检查 `rot.weight`、scale/offset 加载、top-k seed/reuse 与 split cache，不能把 static linear 测试当作 static MoE 支持 |
| Qwen3.5 GDN | 上述 MTP 组合追加 `--spec-tokens 1`、较长 chunk、短长请求混批；检查 conv ring/recurrent state 不串请求、prefill/decode 一致性、spec 接受/拒绝与 draft buffer |
| TP / DP / EP 与 MoE | 在有足够卡的机器上分别追加 `--tp 2`、`--tp 4 --dp 2`、`--tp 8 --dp 8 --ep 8`，各运行 eager/graph；卡数不足则明确未验证。改变 prefill 长度和 batch，覆盖 allgather/all2all/MC2 选择，检查 HCCL 无挂起、共享专家无重复/遗漏 reduction。分派依据见 `lmd_fork/lmdeploy/pytorch/backends/ascend/op_backend.py:515` |
| >1024 专家与 catchall | 仅在有对应 checkpoint 时覆盖 chunked GMM 和 `DLINFER_MOE_PREFILL_CATCHALL=0/1`；检查输出和显存，不把普通 MoE 模型的通过外推到此项 |
| AWQ、int8 KV、multimodal | 用原 A3 格式的 AWQ checkpoint；KV quant 按原模型配置设置并单列结果；multimodal prompt 可用 `--prompts` JSON 传入；检查输入预处理、vision helper 和 mRoPE。没有这些资产时列为未验证 |
| Sampling、profiler、cache copy | 分别跑上述 bad_words、logprobs、随机/greedy 混合 rejection、NPU profiler；cache copy 有实机测试 `lmd_fork/tests/pytorch/ascend/hardware/test_native_cache_copy.py:9`。继续验证大池内存占用；小张量拷贝测试不能证明生产规模无 OOM |
| 迁移前后等价 | 在独立旧环境（基线 LMDeploy + 上述 dlinfer commit）保存同 checkpoint/配置/seed 的 logits/token trace，再与新环境比较。旧环境仅用作 reference，不安装到新环境；新脚本对 Ascend 会拒绝 dlinfer，因此旧基线 trace 须用旧环境的 pipeline harness 生成 |

## NVIDIA GPU 回归

在独立 CUDA 环境安装本分支与对应 CUDA 依赖；不安装 torch_npu 或 dlinfer。使用相同 GPU、driver、torch、依赖 lock、模型、seed、TP 配置分别测试基线和迁移分支；仅比较不同环境的文本不能证明数值等价。

```bash
# PyTorch-only 安装；如需 TurboMind，按仓库原 GPU 构建流程另行编译。
LMDEPLOY_TARGET_DEVICE=cuda DISABLE_TURBOMIND=1 python -m pip install .
python -m pytest -q tests/pytorch/config tests/pytorch/engine/test_cache_engine \
  tests/pytorch/test_cuda_communicator.py tests/pytorch/spec_decode/test_glm_moe_dsa_mtp.py \
  tests/pytorch/spec_decode/test_cudagraph_strategy.py tests/pytorch/models/test_qwen3_5_norm.py \
  tests/pytorch/nn/test_nsa.py
python -m pytest -q tests/pytorch/kernel/test_apply_rotary.py \
  tests/pytorch/kernel/test_rms_norm.py tests/pytorch/kernel/test_fused_moe.py \
  tests/pytorch/kernel/test_paged_attention.py tests/pytorch/kernel/test_mla_attention.py \
  tests/pytorch/kernel/test_causal_conv1d.py tests/pytorch/kernel/test_gated_delta_rule.py
mkdir -p validation/cuda
python tools/ascend/validate_pipeline.py --device cuda --model /models/DENSE \
  --eager --output validation/cuda/dense-eager.json
python tools/ascend/validate_pipeline.py --device cuda --model /models/DENSE \
  --output validation/cuda/dense-graph.json --compare validation/cuda/dense-eager.json
```

继续对 CUDA Qwen3.5/MTP、GLM/MTP、AWQ、InternVL vision 输入运行同样的 before/after 和 eager/graph 比较，重点检查 shared builder、weight loader、GDN norm dispatch、MoE factory、graph 子类和 vision compilation 的变化。支持 FA3 的 GPU 再运行 `tests/pytorch/kernel/test_fa3_attention.py`；不支持的测试保留 skip 原因。需要跨分支运行时，可把 `tools/ascend/validate_pipeline.py` 复制到独立验证目录，使用该分支环境中的已安装 LMDeploy，避免 `PYTHONPATH` 指向另一份源码。

验收记录应保存版本清单、命令、模型 revision、成功/失败/skip 数、trace 和异常栈。本分支已有上述代表性 A3 实测；CUDA、其他模型/量化和多机等未覆盖项仍须独立完成验证。
