# A3 迁移对比：第一阶段完整记录
> 历史记录：原始运行使用 UTC 时间，目录按 2026-11-12 命名。
> 原文中的“未推送”描述当时状态。Mooncake 的旧阻塞判断已由 [第二阶段实测](mooncake_validation.md) 取代；当前决策和分支请看 [总览](README.md)。

工作区：`/root/workspace/a3-validation-20261112`

本轮已建立可复用的迁移前基线，并完成代表性模型、数值、算子、缓存、长上下文和 HTTP 性能对比。不能据此整体签署“迁移无回归”：原迁移分支有一个已修复的启动回归；MTP 长生成的严格 token/logprob 等价未通过，且基线自身也不稳定；若干基线算子和输入评分功能在当前镜像中失败。通过项、失败项和未验证项分别列在下文，不以能输出文字或退出码 0 代替精度验收。

分项结论：Agents-A1和Sage关闭MTP的已测精度、实际prefix复用及两模型261845-token检索通过；Sage的MTP长生成严格等价未通过。当前性能状态请看[最新复核](a3_warm_performance_20261118.md)；第一阶段性能叙述已移至old/，未验收项仍在新主表中保留。Mooncake0.3.13.post1已安装，但两个模型的新旧四个实际启动场景均受fabric内存映射错误阻塞，外部缓存写读、命中、精度和性能未验证。

## 实际比较版本已经固定

| 对象 | commit | 用途 |
| --- | --- | --- |
| LMDeploy main | fb5de87a45df9999a19a569ed9441df8edfa6975 | 迁移前基线 |
| dlinfer main | 3b9096341a41e9d067104a39d914e47ca2d7ec50 | 配套基线 |
| LMDeploy 原迁移分支 | 39dbaa841080d9b9b6b0326197b9de7fe9495a1b | 原始迁移，启动失败证据保留 |
| LMDeploy 本次修复后 | 6421cefea94edd350e419f556158639556357a27 | 后续模型、精度、性能实测版本 |
| dlinfer 清理分支 | 05ede38ae879347be33c8f6ebb136d84f3f4ee88 | 独立功能验证，未修改源码 |

检查时两个 main 正好分别是对应迁移提交的直接父提交，与迁移说明中的基线一致。修复在 LMDeploy 本地独立分支 codex/ascend-a3-validation-20261112；未推送远端。原 /root/workspace/lmdeploy、dlinfer 和同事 /root/workspace/code 下的 checkout 保留。各原 checkout 的 HEAD、已有 diff，以及外部 kernel 的 173 个文件哈希复核见 inventory/preservation-final.json。远程跟踪引用变化不参与比较；实际 HEAD、diff 与模型 manifest 才是本次依据。

## 环境与可比范围

用户确认的镜像为 registry2.d.pjlab.org.cn/ccr-deeplink/lmdeploy_dlinfer:0.18.0-20261102T080655Z。镜像标签不是镜像 digest；容器内实际软件另行记录。机器为 aarch64、320 CPU、约 2 TiB RAM，可见 8 个 64 GiB NPU chip，SoC 251，驱动 25.5.6。当前实际 CANN 为 9.1.0，Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4、Transformers 5.14.1、Triton 3.5.0 / triton_ascend 3.2.2。NPU 计算由 torch_npu 执行，torch 包名中的 +cpu 不代表本次模型在 CPU 推理。

三个不继承 system-site-packages 的 venv 分别用于 baseline、native、standalone。第三方依赖从镜像复制为固定副本，再链接给三个环境，项目安装命名空间独立。没有升级或卸载镜像原环境。native 中没有安装 dlinfer 包，也没有把 dlinfer 源码放入搜索路径；sitecustomize 在主进程及子解释器阻止并记录任何 dlinfer 导入尝试。运行期记录 dlinfer_modules=[]，审计无禁止导入事件。standalone 没有安装 LMDeploy，MoE eager/真实 NPUGraph 运行后也没有 LMDeploy 模块。

外部 triton-ascend-kernels 固定为 8e70a44afa537fcbcdc6d59e6d2196a38f1e7b6f 加同事原有 5 个文件的修改；使用独立源码快照，不直接跟随同事目录。它仍是显式外部依赖，并未冒充已迁入 LMDeploy。补丁、源码哈希见 inventory/code-triton-ascend-kernels.json 与 external-kernel-files.sha256.json。

当前镜像超出两个分支部分依赖声明：torch/torch_npu 要求 <2.10、torchvision 要求 <0.25，dlinfer 的 Python 要求 <3.12。安装使用 --no-deps --no-build-isolation，dlinfer 另加 --ignore-requires-python。三个环境 pip check 均非零，还包括镜像预有的 numpy、profiler 等冲突。原始输出在 inventory/*-environment.json。本轮是对用户指定镜像的受控比较，不能声称完成了符合声明约束的干净安装，也没有擅自修改依赖上界。

主要模型是同一份 /afs-weights/Sage-A1.5 BF16 Qwen3.5 MoE/GDN 权重；补充模型为 /afs-weights/Qwen3-8B dense GQA。用户随后追加 Agents-A1 和两个主模型的 Mooncake 验证，扩展判据及当前状态见文末。模型文件初始 SHA256、大小、mtime、配置与 tokenizer 清单在 inventory/model-*.json；结束时大小和 mtime 不变。Qwen3.5-0.8B 只建立清单，没有实际模型推理覆盖。

Sage 主配置为 TP2、DP1、EP1、BF16、block128、max_batch16、session262144、cache_fraction0.95、prefill8192；部署模式使用 MTP3、prefix cache、state budget16、save alignment128、last-only。单变量或专项变化在各结果 engine_config/options 中保存。dense 补充为 TP1。主对比依次使用逻辑设备 0,1；长上下文边界对比两边均使用 2,3。逻辑 0..7 对应物理 0,1,8,9,10,11,12,13，不混用两类编号。

相对部署文档的偏离均对两边一致：文本推理 language_model_only=True；未运行四实例 router；文档旧参数 --speculative-disable-prefix-cache-block-drop 在两个比较版本均不存在，因此两边都省略；HCCL 建连超时由 7200 缩为 180 秒。完整配置见 inventory/deployment-deltas.txt、logs/*.command.json 和结果 JSON。

比较前判据保存在 ACCEPTANCE.txt：同配置 greedy token 必须 100% 一致；共同历史下 logprob 最大绝对差 <=0.02、RMS <=0.005，argmax 一致；新异常、挂起或状态串扰不通过。性能预热后至少 3 轮，重复出现 >5% 的吞吐下降或延迟上升为回归；波动需复测。部署 SLO TTFT<=3s、输出>=40 token/s 另列，不能从短提示外推所有长度。没有为了让 MTP 通过而放宽阈值。

## 已确认并修复的迁移问题 M1

原始 native 在 Sage MTP3/prefix 的预热阶段报 TypeError: 'module' object is not callable，位置为 AscendFusedMoEImpl.forward。先导入 MoE metadata 子模块后，Python 将同名 package.fused_moe 属性设为 module，覆盖延迟导出的函数名。baseline 可运行同一场景，因此这是明确的迁移启动回归。

修复将 8 个 Ascend backend 文件的 callable 导入改为直接指向定义它们的子模块；新增冷启动和先导入子模块两种顺序的回归检查。未修改算子计算体、算法、图策略或其他厂商行为。修复提交 6421cefea94edd350e419f556158639556357a27，9 文件 48 行增加 / 9 行删除。原失败 logs/native-sage-mtp3-prefix-full.log，修复后完整运行 native-sage-mtp3-prefix-full-importfix。补丁与说明在 artifacts/repair-M1-commit.patch、repair-M1.txt。修复后原生 CPU contract suite 33 passed，修改文件静态检查与 git diff --check 通过。

模型与精度结果如下。表中“通过”只指明确列出的配置/场景；分数比较仅使用共同 token 历史，在首个 token 分歧位置停止，之后的不同历史分数不混入数值判断。

| 场景 | 实际结果 | 判定 / 证据 results/ |
| --- | --- | --- |
| Sage TP2 eager，MTP关，混合长短请求/重复请求 | 21/21 序列、1344 token 一致；6720 个 top-k 分数差为 0 | 通过；compare-sage-eager-score-trace.json |
| Sage TP2 graph，prefill256，MTP关 | 21/21 序列、1344 token 一致；6720 个分数差为 0 | 通过；compare-sage-graph-chunk256-score-trace.json |
| Qwen3-8B TP1 eager | 21/21 序列、1344 token 一致；6720 个分数差为 0 | 通过；compare-dense-eager-score-trace.json |
| Sage graph+prefix，MTP关 | 16 个生成响应一致，12/12 精确答案；另 4 个固定历史评分请求共 123 行 / 640 分数差 0，top1 无差异 | 通过；compare-sage-graph-prefix-extra.json |
| Qwen3-8B graph+prefix | 16 个生成响应一致，12/12 精确答案；另 4 个固定历史评分请求共 123 行 / 658 分数差 0，top1 无差异 | 通过；compare-dense-graph-prefix-extra.json |
| Sage graph+MTP3+prefix 专项 | 16/16 响应、56 token、280 分数完全一致；12/12 精确答案 | 此专项通过；compare-sage-mtp3-prefix-quality.json；明确跳过已另测失败的输入评分 |
| 上述三组 prefix 专项 | 7490 输入 token；首次 cached=0，插入两个干扰请求后重复 cached=7424，检索与重复 token 一致 | 实际缓存命中通过，不仅是配置开关打开 |
| Sage MTP3 近 session 边界检索 | 实际输入 261845 token，输出 8 token，ORCHID-8246 正确；40 分数差 0；耗时基线44.376s / native44.421s | 单条边界检索通过；compare-context-boundary.json；不是全长并发压力认证 |
| Sage MTP1 graph 长生成 | 17/21 序列一致；共同历史分数 max1.562498、RMS0.139313 | 严格等价失败；compare-sage-mtp1-score-trace.json |
| Sage MTP3 graph 长生成 | 13/21 序列一致；max2.437459、RMS0.147461 | 严格等价失败；compare-sage-mtp3-score-trace.json |
| Sage MTP3 eager 长生成 | 15/21 序列一致；max1.454253、RMS0.145838 | 严格等价失败，不仅在 graph 模式出现；compare-sage-mtp3-eager.json |

非 MTP 的 eager 与 graph/prefill256 跨模式比较在两个分支上都是 19/21 一致，差异是同两个请求、同位置、同 token；同模式的新旧比较均一致。此对比同时改变 graph 和 chunk，不能将其定位成单一 graph 问题，也不能归为迁移新增。见 cross-mode-audit.json。

## MTP 差异的重复性控制

| 独立进程，所有 Python/NumPy/Torch 及 worker 固定 seed42 | 完全一致序列 | 共同历史分数 max / RMS |
| --- | --- | --- |
| baseline 自身 r0 对 r1 | 14/21 | 1.468748 / 0.138434 |
| native 自身 r0 对 r1 | 13/21 | 1.618086 / 0.144442 |
| baseline 对 native，r0 | 15/21 | 1.867675 / 0.148638 |
| baseline 对 native，r1 | 16/21 | 1.394611 / 0.143870 |

这些控制证明旧基线本身在 MTP 长生成中不满足严格重复性，不能把每个新旧 token 差异直接认定为迁移造成；它们也不能证明迁移没有精度回归。本次结论保持“严格验收未通过、归因未完成”。原始 first_mismatches 含首个位置与双方 top-k 分数；MTP1 可见 token 分歧前已有分数漂移，MTP3 也有接近并列最高分的分歧。没有通过强行放宽阈值、改采样算法或只比最终文本掩盖这些问题。固定历史输入评分原本用于进一步区分，遇到下面的共同崩溃。

主部署配置最初 128-token full trace 是 10/21 序列一致，但当时默认 logprobs_mode=None，未取得有效分数；该数值部分明确为 unverified-no-scores。随后重新使用 raw_logprobs 得到上述结果。早期名义 260000 长度的提示实际只有 236225 token，且 ignore_eos 强制尾部生成出现差异，未将其计为 256K 通过；后面的 261845-token/EOS 边界测试才是表中依据。

## 基线自身与环境共有问题单独记录

1. Paged MLA prefill、paged MLA graph replay、MLA decode 三项在 baseline、native、standalone 全部以同一 keyRope 非连续错误失败：CANN FusedInferAttentionScoreV4 error561002。普通 MLA prefill 的 reference test 通过不代表 paged/decode 已通过。证据 operator-comparison.json 与三份 JUnit XML。未擅自增加 contiguous 拷贝修改算法或性能。
2. Sage MTP3 固定历史评分请求 max_new_tokens=0 / logprob_start_len=0，在两边共享的 strategies/ar_spec/sequence.py 路径因 len(draft_token_ids=None) 崩溃。两边在此之前 12/12 精确答案通过，之后评分响应失败、engine 不可用，因此原 extra 运行 complete=false。logs/{baseline,native}-sage-mtp3-extra.log 和对应 JSON 保留。后续 prefix 专项明确 skip-input-scoring，不能把专项完成解释成此错误已修复。
3. MTP 长生成重复性不满足严格判据，见上述同版本固定种子重复控制。
4. pip check/版本约束失败属于当前镜像共同环境限制，详见环境段。

## 算子和清理后的 dlinfer 独立验证

| 测试 | 通过 | 失败 | 解释 |
| --- | --- | --- | --- |
| native CPU contract，修复后 | 33 | 0 | 含新增导入顺序两例；不是 33 个 NPU kernel |
| standalone 清理边界 CPU contract | 16 | 0 | framework 导入边界及原有 helper 行为 |
| baseline 算子套件 | 25 | 3 | 16 个 CPU conv-ring reference + 9 个真实 NPU 通过 |
| native 硬件目录套件 | 26 | 3 | 16 个 CPU conv-ring reference + 10 个真实 NPU 通过 |
| standalone 算子套件 | 25 | 3 | 16 个 CPU conv-ring reference + 9 个真实 NPU 通过 |
| standalone MoE 独立 CPU reference 对照 | 6 | 0 | 1056 专家，N32、hidden/intermediate64、BF16；4 种 eager catchall/type + graph capture/replay 更换 routing；max差0 |

真实 NPU 通过项包括 standard/dense/普通 MLA prefill、3 种长度的 interleaved RoPE、Lightning Indexer、Sparse Flash Attention、split MLA KV 写入；native 另外通过小池 cache copy 与 workspace reuse。具体 case 名和断言在 XML，不能把 16 个 CPU ring reference 算作实际 Triton decode 验证。Sage 模型链路实际执行了 GDN/conv decode，但这仍不是所有 shape 的完整 kernel 数值证明。

standalone-moe.json 证明清理后的 dlinfer 无 LMDeploy 安装/导入时，>1024 专家的 eager 与真实 NPUGraph 分支可用；小维度合成数据不代表 1056 专家完整 checkpoint 的吞吐或容量。独立 Transformers remote-module/DynamicCache、DICP/ATB compiler 不在这项验证范围。

默认 Torch rejection（baseline/native）与 native 可选 Triton rejection 各自通过 8 种 greedy 接受前缀 pattern，以及 65536 个纯随机、32768 个混合批随机样本的首 token 分布检查。理论分布 [0.1,0.2,0.3,0.4]，预设误差 <=0.015；两边 Torch 频率完全一致，Triton 也在阈值内。证据 *-rejection-reference.json。此项只覆盖 vocab4、draft_probs=None 的构造输入，不能替代可选 Triton 模式下完整模型、任意 draft distribution 或 bad-words 的端到端验证。

Sage的最新真实HTTP结论：在两侧匹配的正确编译器上，当前PR的C1总吞吐、流式速度和TPOT通过±5%等效判据；C4和TTFT仍有不确定项。[完整新结果](a3_warm_performance_20261118.md)。旧错误编译器下的数字及SLO测量见[历史归档](old/migration_validation_before_warm_20261118.md)，不把它们改写为只靠预热修复。

## 仍未验证或不能外推的范围

- GPQA 等标准数据集准确率、部署文档的 GPQA 86.9 指标；12 道确定答案题只是可复现的有限检查。
- DeepSeek-V2/GLM DSA/ChatGLM 完整 checkpoint、GLM MTP、ModelSlim W8A8、AWQ、int8 KV、其他量化；已有小算子测试不等于这些模型通过。
- DP/EP、多机、MC2/all2all/full通信组合、四实例 router 和满卡部署；只实测了 Sage/Agents TP2 与 dense TP1。
- 完整模型 random/greedy 混批、bad-words、可选 Triton rejection 的端到端路径；分布小测试的边界已单列。
- NPU profiler trace、生产规模 KV-copy/OOM 压力、所有 graph lifecycle/并发取消/重连恢复、长期稳定性。
- vision/multimodal、InternVL helper、CUDA/TurboMind、MACA/CAMB/PPU、A2/310P；没有这些路径的实机通过结论。
- 受支持 Python/torch/CANN 组合下的重新干净构建、A3 wheel 安装验收；本次项目是隔离 editable 安装。
- 完整固定历史 logits/KL 或 perplexity。保存的是 top-k logprob 和精确 token，对 MTP 的输入评分路径还遇到共有失败。

## 复现与证据入口

REPRODUCE.txt 提供可直接使用本目录环境重跑的命令。每次使用新的 label/output，避免覆盖旧运行。logs/*.command.json 保存具体 argv、环境差异、开始结束时间、退出码；较后运行还记录 source HEAD/diff 和当次 harness SHA256，对应副本在 artifacts/harness-snapshots/。results 保存原始 prompt、token IDs、top-k logprob、缓存命中、SSE 文本和时间；XML 保留失败和跳过，results/summary.json 与 numerical-comparisons.csv 为汇总索引。inventory 保存参考文档、安装脚本、软件环境、模型/外部 kernel 指纹。修复提交及 format-patch 在 artifacts。

最初辅助检查把物理 8/9 当作容器逻辑设备，造成失败或硬件 skip，已改用逻辑 2/3 重新执行；那些旧记录保留但不计通过。HTTP 早期连接池客户端的一次 ServerDisconnectedError 和未完成批次也保留；服务在该异常之前未自行退出，断开后退出是 harness finally 的 SIGTERM。客户端空闲 keepalive 的竞态是可能原因，未将其冒充已定位的 LMDeploy 缺陷；正式比较两边统一每请求新连接、无自动重试。

本次可以交付固定环境、可执行基线、修复后的独立原生环境和分项验证证据；尚不能给出完整生产发布或全模型无回归认证。明确回归 M1 已修复，其余 MTP 等价失败及共有基线缺陷的后续处理，应继续保持同配置对照并独立立项，避免混入算法优化。


## 用户追加的 Agents-A1 与 Mooncake 0.3.13.post1 验证

扩展判据在执行前写入 ACCEPTANCE-EXTENSION.txt；两份追加的部署文档已复制到 inventory。Agents-A1 使用 /afs-weights/Agents-A1 的同一份 BF16 权重、TP2、session262144、block128、batch16、prefix budget16/alignment128/last-only。按其 A3 文档关闭 MTP；Sage 部署对比保持 MTP3。Agents 的完整模型 SHA256/配置清单已保存，结束时文件大小和 mtime 不变。

| Agents-A1 场景 | 实际结果 | 判定 / results/证据 |
| --- | --- | --- |
| eager 长短请求与重复请求 | 21/21 序列、1344 token、6720 分数完全一致 | 通过；compare-agents-eager.json |
| graph+prefix 质量与固定历史评分 | 16 个生成响应/56 token/280 分数一致，12/12 确定答案；4 个评分请求123行/635分数差0，top1无差异 | 通过；compare-agents-graph-prefix-extra.json |
| 真实 prefix 复用与干扰请求 | 7490输入token，重复命中7424token，插入两个其他请求后检索和输出仍一致 | 通过；*-agents-graph-prefix-extra.json |
| 近上下文上限检索 | 261845输入token、8输出token、40分数一致，均得到ORCHID-8246；单次36.585s/38.201s | 此单条场景通过；compare-agents-context-boundary.json |

以上覆盖实际数值、缓存状态和边界长度，不代表完成 GPQA 82.3 指标、全部 agent/tool-use 任务或多模态精度。短题检查和单条长上下文不能外推到完整模型评测。

Agents的当前PR HTTP C1总吞吐和TPOT已通过等效判据；C4、TTFT和旧依赖历史对照仍有不确定项，继续定位。[完整新结果](a3_warm_performance_20261118.md)。旧HTTP表格和固定批次pipeline数据见[历史归档](old/migration_validation_before_warm_20261118.md)。

## 第一阶段 Mooncake 失败配置（已被后续复核取代）

Mooncake 使用指定的 mooncake-transfer-engine-npu==0.3.13.post1，已安装到新增 baseline-mooncake/native-mooncake 两个环境。Python3.12/aarch64 wheel 为15706817字节，SHA256 ba134f2cc99784aa32404c3a1406e3c85400792fb2c62bb458cd4757a07cc4bb。原三个环境仍保留镜像的0.3.11.post1，未被升级。新增 native 环境也没有安装或导入 dlinfer，审计见 inventory/*-mooncake-environment.json。

按 GLM/H200 文档采用 embedded MooncakeStoreConnector、kv_role=kv_both、P2PHANDSHAKE 和单独 master。A3 传输选 protocol=ascend，开启 ASCEND_ENABLE_USE_FABRIC_MEM=1；使用本次管理的127.0.0.1:55051/55052服务，没有连接 /afs-weights/mooncake.json 中的外部 master。A3 验证配置将每worker全局segment设为2GB、local buffer设为256MB，其余模型配置与各自部署对比一致。该容量用于代表性功能验证，不能等同H200文档160GB/4GB部署的容量或性能。

发现两边原代码都拒绝 protocol=ascend，仅允许rdma/tcp。这是共有平台接入缺口，已在额外分支允许ascend并验证配置原样传给SDK。两个环境配置/初始化测试各26 passed、77 deselected，属于CPU/stub测试。baseline额外提交e285fa49d1759d709660588764ce4091566240af，native额外提交9f76d08a26b484b5bd23391ffb829a9f50c4f635；原main/native比较工作区不变。补丁只改协议校验及参数传递测试，没有更改kernel或传输算法；不能用这项通过替代真实数据传输。

| Mooncake 实测 | 结果 | 归类 |
| --- | --- | --- |
| NPU wheel安装与API导入 | 新旧扩展环境均为0.3.13.post1 | 通过，安装事实 |
| 独立SDK、两个NPU进程、fabric开启 | 两个环境均在Store.setup失败；64MiB segment/16MiB local尚未加载模型就出现aclrtMapMem错误及std::bad_alloc | 共有运行环境阻塞；零次有效数据写读 |
| 设置fabric变量为0的控制 | 仍显示use_fabric_mem=true，同样失败 | 该版本只判断变量存在，0不等于关闭 |
| 真正unset fabric变量的SDK控制 | ADXL初始化失败503900，无法从hccn.conf/hccn_tool读取NPU IP | 另一项环境阻塞；不计非fabric传输通过 |
| Sage MTP3+Store，baseline/native | 两边实际加载模型并进入缓存引擎构建，Store.setup的fabric映射失败，rc1 | 2个启动场景失败，精度/命中/性能未验证 |
| Agents MTP关闭+Store，baseline/native | 同一Store.setup的fabric映射失败，rc1 | 2个启动场景失败，精度/命中/性能未验证 |

四个模型运行均保存 complete=false 的结果和完整栈，见 results/mooncake-model-progress.json、*-fabric-model.json 和 logs/*-fabric-model.log。计划中的跨引擎外部cache hit、NPU缓冲区写读一致性、GDN状态恢复、缓存污染隔离与Store冷/热性能全部未完成，不作通过判断。错误在小内存、无模型的SDK探针中也存在，不能解释为迁移新增的模型OOM。

该版本的官方Ascend Direct文档要求fabric memory模式使用CANN9.0+、HDK26.0+；当前CANN9.1.0，而npu-smi board实际报告Software Version25.5.6、Firmware7.8.0.10.220。实际驱动映射失败与这项版本缺口同时存在；不能仅凭此错误断言升级驱动是唯一所需修复。容器内也没有找到/etc/hccn.conf或hccn_tool，npu-smi的IP查询明确不支持；未猜测或修改NPU网卡地址，已向用户询问是否有其他挂载路径。当前证据不足以继续验证Store数据传输。

本次NPU wheel实际走Ascend Direct/HIXL，日志明确显示fabric选项。镜像已有memfabric_hybrid1.2.0，但没有把“该包已安装”当作fabric传输已通过。单独的mooncake-transfer-engine-memfabric在查询时只有0.3.14，没有擅自替换用户指定的0.3.13.post1。源代码还确认环境变量只要存在就打开fabric，关闭必须unset。官方依据及固定commit链接在 inventory/mooncake-reference-sources.txt，wheel hash与源码快照已保存。

继续完成Mooncake验证需要宿主提供可用的fabric内存映射能力和匹配的驱动/运行时，或先补齐真实NPU网卡配置以验证非fabric路径。补齐后应从独立SDK的实际NPU写读校验开始，再验证两个模型的跨引擎Store命中及同配置精度/性能；本次不将这些尚未执行的后续步骤描述为成功。

结束审计：88份运行命令记录均已有退出码；本次测试进程及25485/55051/55052监听服务全部退出，npu-smi未发现运行中的NPU进程。原工作区、同事改动和固定比较源码复核通过，native及native-mooncake均无dlinfer安装痕迹，禁止导入事件为0。证据见inventory/cleanup-final.json、preservation-final.json。环境和源码保留在本机；evidence.tar.gz归档日志、数据、脚本、补丁、wheel与外部kernel快照，EVIDENCE-SHA256.json逐文件记录校验值。
