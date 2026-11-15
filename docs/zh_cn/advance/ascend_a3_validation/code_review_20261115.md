# 2026-11-15 测试边界与公共接口审查

日期：2026-11-15，Asia/Shanghai。对应实现提交：
[`c3947a93`](https://github.com/silencelamb/lmdeploy/commit/c3947a93c786876e17bb64dfb6f61ee9f1e7b221)。
本次修正测试收集边界和平台名称，不改变 kernel 计算体；上游迁移 PR 提交流程仍暂停。

## 测试来源和 CI

六个 `test_ascend_*.py` 文件来自 dlinfer
`3b9096341a41e9d067104a39d914e47ca2d7ec50`：attention precision、causal-conv ring、
interleaved RoPE、sparse attention、graph lifecycle、graph NSA metadata。
迁移保留其参考计算和断言，修改 import、GraphRunner 接口及必要的 runtime mock。

`test_native_integration.py` 是迁移特有的 CPU 契约测试；其中小型混合 dtype cache
分配场景承接旧 `test_lmdeploy_cache_allocator.py` 的测试目的，并按新分配接口重写。
`test_native_cache_copy.py` 和 `test_kernel_import_order.py` 分别验证新增原生缓存拷贝及
本次 A3 发现的导入顺序回归。真实模型数值、长上下文、HTTP 和性能测试属于独立验收 harness。

现有 `.github/workflows/unit_test.yml` 在 A100 runner 上执行 `pytest tests`。
原迁移没有完整排除 Ascend 硬件目录，真实无 torch_npu 的环境中，cache-copy 测试的
autouse fixture 会因导入 torch_npu 失败。这是本轮发现并修复的测试隔离缺口。

修正后，公共收集在导入测试模块前排除 `tests/pytorch/ascend/hardware`；直接选择该路径
却未显式开启时，给出清晰错误。CPU 契约继续在公共 CI 运行，导入顺序测试在没有 torch_npu
时跳过。Ascend 硬件 suite 使用：

```bash
python tools/ascend/check_runtime.py --require-kernels
python -m pytest -q tests/pytorch/ascend/hardware --run-ascend-hardware
```

可以把这两条命令配置到独立 Ascend runner；本轮没有注册 runner 或新增自动运行的 Ascend
workflow。GPU CI 仍应覆盖公共模型/配置改动。ring 测试虽然在硬件目录，但计算使用 CPU
reference，并依赖 Ascend Triton 模块导入，不能计作 NPU convolution 数值通过。

## default/gated_norm.py

原先 `nn/gated_delta.py` 直接构造 FLA `FusedRMSNormGated`。现在通过
`RMSNormGatedBuildSpec` 选择实现：CUDA 继承默认 FLA builder，Ascend 提前处理相同
BuildSpec 并构造迁入的 Ascend `RMSNormGated`。默认文件保留原构造逻辑和延迟导入；
没有引入另一套归一化算法。独立小文件是实现组织方式，并非计算上必需。

## compile_for_cuda

这是 InternVL / InternVL3-HF 的四个 pre/post RMSNorm helper 使用的局部编译策略。
当前 device context 为 CUDA 时，首次调用创建并缓存 `torch.compile` wrapper；Ascend
等其他 context 直接执行原函数。两种平台均可调用该 wrapper，但只有 CUDA 在这里启用
torch.compile。它承接旧 dlinfer 关闭这些 helper 编译的行为，避免临时替换全局
`torch.compile` 并覆盖模型函数。它与 NPUGraph capture/replay 是不同层次的机制。

## ascend 与 npu

LMDeploy 平台标识统一为 `ascend`：backend selector、模型配置、runtime 入口、is_ascend
及 dispatcher 不再把 `npu` 当作 Ascend 别名。原公共 EngineConfig 本身只接受 `ascend`。
底层 `torch.npu`、`torch.device("npu")` 和 tensor 的 `device="npu"` 保留，它们是
torch_npu 的设备接口名，不能机械替换为 `ascend`。

## cudagraph.py 的复用范围

共享的是模型的 `CudaGraphMixin` / `CudaGraphMeta` 契约。四个入口按平台显式分派：
`support_cuda_graph`、`make_buffers_cudagraph`、`fill_buffers_cudagraph`、
`update_context_cudagraph`。独立 `AscendGraphRunner` 仍负责 `torch.npu.NPUGraph`
capture/replay/update，Ascend helper 处理自己的 buffer dtype、CPU metadata、GDN/MTP
状态与 DSA cache。并非直接复用 CUDA graph runtime。

runner 通过模型方法调用这些接口，保留 MTP 子类的 `super()` 链及额外 hidden-state buffer。
公共入口改动较小，是因为保留了模型接口，同时迁入了原有 Ascend graph 实现。

## 本轮验证

| 检查 | 结果 | 边界 |
| --- | --- | --- |
| 修复前 cache-copy，独立无 torch_npu 环境 | 1 个预期 fixture error | 复现 GPU/CPU 环境误收集时的依赖错误 |
| 修复后默认 Ascend 测试目录，无 torch_npu | 32 passed，1 skipped | skip 为 vendor import 测试；硬件目录未收集 |
| A3 vendor 环境默认 Ascend 测试目录 | 34 passed | CPU 契约及两种 kernel 导入顺序，不等于整模型推理 |
| 从 `pytest tests` 根入口收集 | 32 个 CPU case | 显式排除其他无关套件，验证父 conftest 晚加载时同样排除 hardware |
| 直接选择硬件文件但不开启 opt-in | exit 4，明确要求 flag | 未启动 NPU 测试 |
| 显式开启硬件目录收集 | 29 个 case | 只检查 collection，没有重跑全部算子 |
| 真实 NPU cache-copy，显式 opt-in | 1 passed | 检查 opt-in 后仍可执行及缓存池/工作区语义 |
| Ruff 0.15.4、Python AST、git diff --check | passed | 本轮修改文件 |

本机没有 NVIDIA GPU；没有宣称完成 CUDA 实机 CI，也没有重跑整模型、长上下文和性能对照。
原来的 MTP、HTTP C4 观察与共有 MLA 失败记录保持不变。

本轮记录位于本机 `/root/workspace/a3-review-20261115`。
选定命令、日志、JUnit 与源码哈希见 [审查证据](evidence/review-20261115/index.json)。
早期使用 import hook 模拟缺少 vendor 包时触发第三方 accelerate 导入状态问题；随后改用
实际不含 torch_npu 包和 metadata 的独立环境，以上计数取后者。早期诊断日志仍保留在本机。
