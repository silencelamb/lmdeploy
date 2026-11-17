# PR #5134：A3 精度、修复与环境验证汇总

更新日期：2026-11-17，Asia/Shanghai。本文是PR正文A3总表的统一详情入口；整理已有实测，不将历史数据改标为新HEAD上的新运行。

## 结论与比较范围

A3已测的匹配非MTP迁移对照以及gated norm重构对照，记录的token、完整top-k集合和分数逐位一致。Sage-A1.5的circular GDN/MTP状态问题已修复，集成修复后的旧/新编译器环境均完成全新进程重复验证。不是所有模型、MTP配置或生产拓扑都已验证；跨NPU/CUDA逐位一致也不属于本次判据。

| 模型 | 精度结论 | 实际比较范围 |
| --- | --- | --- |
| Sage-A1.5 | 非MTP对齐；修复后的已测MTP3场景稳定并匹配正确参考 | 历史dlinfer→native迁移、native norm重构、circular-state修复后独立回归 |
| Agents-A1 | 非MTP对齐 | 历史迁移及native norm重构；不外推为全部MTP配置通过 |
| Qwen3-8B | 非MTP dense补充对齐 | 历史迁移、eager及graph/prefix补充 |
| Qwen3.5-0.8B-do01 | 非MTP norm重构对照对齐 | 已有native实现的重构前后，不冒称所有旧dlinfer路径均实测 |
| Intern-S2 Preview、Intern-S2 Mobius | 非MTP norm重构对照对齐 | 对应A3 checkpoint；不能等同H200的Preview397B-FP8全部部署配置 |
| InternDecision、Qwen3-Next | 完整A3推理未验证通过 | 两个native比较版本都有启动问题；构造/权重/算子覆盖另列 |
| GLM-5.3-Flash、GLM-5.3 | 完整A3模型未验证 | Flash的真实gated scale/数学参考通过，不替代缺失KDA支持的整模型验证 |

“对齐”是相同硬件、checkpoint、配置和实际比较阶段内的结论。S2与dense Qwen3.5的norm重构数据使用记录的实现快照；不能把不同阶段的SHA统一写成当前PR HEAD。

## 固定版本与环境

| 阶段 | 比较版本 |
| --- | --- |
| 原迁移 | LMDeploy基线fb5de87a45df9999a19a569ed9441df8edfa6975＋dlinfer3b9096341a41e9d067104a39d914e47ca2d7ec50，对native6421cefea94edd350e419f556158639556357a27 |
| gated norm重构 | native342db85ecff3c0dc864a4bf8613c589c270f4141，对5f1b1b3c12b1c198813f65275f778c7ad292fb02及源报告列出的提交前快照 |
| 集成circular GDN修复 | 85a0581228ad33a695430a1f43a47358a8258131；旧/新编译器各两轮完整Sage MTP |
| 安装/CI后续 | 143fa1e6修复sampling合同；111dbf6b/f64完善A3安装；2ccf38a43f0fdc7415ce4ce9f177852bdacc9966为已记录CI完成HEAD |

各阶段环境、源码快照与diff分别保存，不能用后续CI合并预览或当前依赖组合替换历史对照。

历史A3主机为aarch64、可见8个64GiB NPU chip、SoC251，约2TiB内存。实际硬件比较使用CANN9.1.0、Python3.12.13、Torch2.10.0+cpu、torch_npu2.10.0.post4；NPU计算由torch_npu执行。模型以Sage/Agents TP2、dense TP1为主；具体prefill、缓存、上下文及MTP配置见原始命令。

当前约束安装配方固定CANN9.1.0/Python3.12，Torch2.10.0+cpu、torchvision0.25.0+cpu、TorchNPU2.10.0.post4、Triton-Ascend3.2.2.post2/NPUIR2.0.0；Ray2.48.0、Transformers5.14.1、NumPy1.26.4、Pandas2.2.3、PyArrow25.0.0及datasets5.1.0。它与旧镜像的非干净依赖检查是不同记录。完整CANN9.2/驱动组合、全部下游版本范围未验证。

原生父进程和worker使用dlinfer导入守卫。当前应用venv安装、pip check和核心导入检查通过；CANN可选SDK/profiler工具的依赖不计为应用venv依赖。现有用户目录和旧环境保留。安装文档与源码来源见[安装/CI完整记录](ci_docker_recovery_20261116.md)及[约束环境验证](mtp_ray_investigation_20261116.md#ci修复与cann91-docker配套)。

Native Ascend仍需要TorchNPU、CANN/NNAL/ATB和Triton-Ascend，并保留torch_npu显式transfer_to_npu初始化；移除dlinfer不表示厂商兼容层完全没有进程级影响。MACA/CAMB继续使用原有dlinfer路径。

## 模型、算子和精度证据

- Sage非MTP eager及graph/prefill256：每组21响应、1344 token、6720分数一致；Agents对应非MTP对照也逐位一致。真实prefix命中与两模型261845-token检索有独立记录。
- norm重构：Sage、Agents、Qwen3.5-0.8B-do01、Intern-S2 Preview/Mobius各21响应、1344 token、6720分数逐位一致；保存完整top-k集合，非只比交集。Sage/Agents另有256-token长生成30/30响应、7680 token一致。
- norm接口/构造：两树各98通过、61跳过；55真实A3算术/NPUGraph检查通过，32组冻结旧实现逐字节相同。
- 208层真实checkpoint scales：174个SiLU层与旧kernel逐位一致；Flash的34个sigmoid层通过独立数学参考。旧Ascend工厂忽略显式sigmoid/dtype是本次已修复的接口语义问题，不应要求复现其错误SiLU输出。
- 新增CUDA norm文件随后在A100 CI通过83项，含56真实CUDA/graph；8个caller构造/加载检查通过。完整H200结果单列在[GPU报告](cuda_regression_20261116.md)。

原始精度、全部配置及失败见[迁移记录](migration_validation.md)和[norm重构证据](gated_norm_refactor_20261115.md)。

## Circular GDN/MTP：具体问题、解决方式和验证

**问题：**固定seed、greedy下，Sage MTP长生成仍可分叉。首差发生在相同历史和draft IDs下的target状态/logits；实际Triton-bundled NPUIR1.2.0/042c923d自动多缓冲把状态buffer和同步事件绑定到不同循环层级，后续更新可覆盖尚未完成的中间状态写出。当前token输出可能正确，随后MTP读取错误保存状态才出现差异。

**修复：**[85a05812](https://github.com/silencelamb/lmdeploy/commit/85a0581228ad33a695430a1f43a47358a8258131)只将circular路径设为`num_stages=1`，其他路径保留3；保留递推、dtype、采样/接受和回滚规则。新增Q=1/2/4、eager/graph的独立CPU状态oracle，逐个检查全部保存槽。新编译器也保留该安全默认；单缓冲并非所有形状都更快。

上游已发布修复05f2adcf，官方NPUIR2.0.0/efb71415包含它；Triton-Ascend3.2.2.post2携带此编译器。PR提供固定升级配置，但实际加载路径必须核对，不能只看PATH或Python包版本。

| 验证 | 已记录结果 |
| --- | --- |
| 最小多缓冲复现 | 旧编译器100/100状态错误；官方2.0.0为0/100 |
| 真实冻结GDN、新编译器stage3 | 200/200重放状态和输出逐位正确 |
| 集成PR旧/新编译器硬件回归 | 各6/6通过，实际circular启动参数stage1 |
| 集成PR CPU合同 | 32通过、1跳过 |
| 最终Sage MTP3、TP2、graph/prefix | 旧/新编译器各两次全新进程；每轮21响应、1344 token、6720分数匹配正确参考 |

这是已测Sage MTP配置的修复闭环，不扩大成所有模型/MTP/长上下文均通过。旧MTP失败记录保留。具体IR、两种独立因果干预、升级比较和性能形状依赖均在[完整GDN调查及集成证据](mtp_ray_investigation_20261116.md#修复纳入迁移pr)。

## 其他已修复项与CI

- 原迁移callable被同名子模块覆盖：直接从定义模块导入，6421cefe修复并有冷启动/导入顺序回归。
- RMSNormGated规范化：这次补齐1类算子、3个直接模型调用点，将特殊工厂/平台Module改为公共nn层持有权重、typed BuildSpec和无模型参数的后端实现，对齐已有RMSNorm/LayerNorm组织。Ascend显式sigmoid/dtype处理随参数合同补齐，CUDA继续使用原FLA计算。[设计前后对比](gated_norm_refactor_20261115.md#规范化范围一个算子三个直接模型调用点)。
- PR新增sampling分支曾引入3个cache_config失败：143fa1e6改按实际NPU tensor处理，不为测试补无关cache配置。当前固定GPU检查中这3项不再失败。
- CANN9.1安装/镜像对齐并捆绑已验证prefill/L2Norm源码、版权及许可，删除浮动GitCode安装依赖；Jetson单独固定cuDSS ABI并登记动态库目录。
- 2ccf38a4记录的14/14 CI门禁完成：完整unit7452通过、400跳过，ETE有351次pytest通过记录和1次自动rerun；这不是351个唯一测试函数。CI实际checkout合并预览fea0d6fd，不能把历史模型数据重标到该预览。[全部CI任务、环境与失败恢复](ci_docker_recovery_20261116.md)。

## 性能与未完成范围

原Agents norm重构C4吞吐−5.15%等信号在同worker交错对照中未稳定重现；实际旧/新norm NPU二进制相同。现有证据未确认norm因果性能回归，但不能写成端到端性能全部通过。逐轮性能及替换实验首捕获失败保存在[性能调查](gated_norm_performance_20261116.md)。

约半秒主机停顿是独立问题，已有CPU18调度/softirq和纯C++证据，具体回调及永久修复仍开放；详见[独立Ray/主机延迟报告](ray_host_latency_20261117.md)。CPU亲和性、GC关闭和await改序都不是本PR交付的修复。

未通过或未覆盖的A3项目包括：共有MLA keyRope连续性失败；MTP零输出token输入评分；InternDecision普通norm dtype启动问题；Qwen3-Next配置启动问题；GLM完整模型/KDA支持；其他量化、视觉、多机/DP/EP生产拓扑、高并发长稳及标准能力榜单。规范安装profile的通过不追溯抹掉旧环境依赖失败；算子/权重测试不替代整模型验证。
