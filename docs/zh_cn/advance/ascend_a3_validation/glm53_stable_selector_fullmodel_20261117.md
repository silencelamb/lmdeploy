# GLM-5.3 保序选择器：单机8卡完整模型验证

执行时间：2026-11-17 17:22–18:57，Asia/Shanghai。

## 结论

**已有deterministic=True选择器已完成真实完整模型验证。** 在两份固定源码上分别启用该选项，用同一份完整GLM-5.3 FP8 checkpoint、单机8×H200、TP8，完成32次全新进程运行。非MTP/MTP3 × eager/graph × prefix off/on共8组配置，其基线自重复、PR自重复和两次跨源码配对，共32项比较全部通过：生成token、完整top-5集合及逐项分数二进制完全相同。

这是**相同配置、相同缓存事件下**的整模型自重复和迁移对照。prefix开启时，同一版本长请求首次cache miss与随后cache hit的返回top-5集合/分数存在差异，所有生成token仍相同；两个版本在各自对应的miss、hit条件下均固定可复现并逐位对齐。该跨缓存执行路径的数值变化仅作现象记录，**不作为本次迁移回归或验收阻塞，不再展开算子级排查**。同配置、同缓存状态的自重复及迁移对照继续使用零容差。

本轮是独立研究工作树中的控制验证，**没有把保序改动合入迁移PR #5134**，也没有实施Intern-S2的AllReduce修复。原未修改默认选择器的失败和原主矩阵36/44结论保留，不被新控制数据覆盖。

## 实际启用方式与版本

当前选项属于CUDA sparse_index_topk函数，并非PytorchEngineConfig的同名公开配置。本轮仅在真实NSA选择调用处传入该现有参数：

```python
self._sparse_index_topk(..., sorted=False, deterministic=True)
```

没有更换选择算法、排序参考结果代替GPU执行、修改sampling、放宽阈值、改变同步或使用CPU绑核。内核实现本身未修改。

| 对象 | 未修改父提交 | 本机研究提交 |
| --- | --- | --- |
| 基线＋稳定选择控制 | 442097e83799c2e85df9d6bbce35c7c79ea05dfe | f06316de64fe8d3c23f72d17bd94180965a4558e |
| 固定PR＋稳定选择控制 | f64ab5061153b5c5b63721cf2d8d6f1b41b95104 | 5e51b6718ec31dbcbdbc5fc8092a71b3294e710d |

两个研究提交都只改`lmdeploy/pytorch/backends/cuda/nsa.py`中的一个调用。工作树为`worktrees/glm-det-base`、`worktrees/glm-det-pr`；分支为`research/glm53-stable-base-20261117`、`research/glm53-stable-pr-20261117`。原始base、pr-f64和迁移PR分支未修改。研究提交及patch用于复现控制，不代表已完成按模型配置接入的生产修复。

所有32次运行均记录干净的研究提交、实际导入路径和CUDA依赖。256份worker选择器审计（每轮8份）确认导入所选工作树、K=2048支持且实际调用处明确传True。两份[复现patch](evidence/glm53-stable-fullmodel-20261117/glm-det-base.patch)、[PR侧patch](evidence/glm53-stable-fullmodel-20261117/glm-det-pr.patch)已发布，可应用到固定父提交；不依赖研究提交已推送到远端。

## 环境和完整工作负载

- GPU为原8块H200，TP8、DP1/EP1；BF16激活、官方已有FP8权重，不量化转换、不替换小模型。GPU入口锁和每轮启动前显存检查防止本任务内部重叠。
- checkpoint为`models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516`：141个权重分片、755617140416 bytes。150个运行文件的大小、mtime和symlink目标与原全量hash记录一致，非权重文件另外重新计算SHA；权重SHA复用原始完整读取记录，未冒称又全量重哈希一遍。
- Python3.12.14、Torch2.10.0/CUDA12.8、Triton3.6.0、TileLang0.1.11、FLA/fla-core0.5.2、Transformers5.16.0、NumPy2.2.6、Ray2.58.0、NCCL2.27.5；两环境实际Torch路径指向同一依赖快照。
- 沿用保存的匹配Triton cache。非MTP固定4276个GPU blocks，MTP3固定4097，与对应旧运行实际容量相同；block64、max batch8、session65536、prefill2048。
- 固定Python/NumPy/Torch/CUDA seed42，greedy、do_sample=False/top_k=1。每轮包含单请求、8请求批次、长短混批、单请求与批请求新请求重放、32603-token长请求及重放。
- 每轮23响应、1408生成token、7040个完整top-5分数；短请求输出64 token，两个长请求各32 token。比较原dtype返回值的二进制表示，零容差，不只比较文本。
- 两侧各两次全新进程，顺序base0→PR0→PR1→base1；先完成非MTP，再开始MTP3。父进程和worker均使用NPU/dlinfer导入守卫。未修改或使用A3 CPU18亲和性规避。

这里的32次运行实际累计736个响应实例、45056个生成token、225280个top-5数值；重复运行重复使用同一批prompt，不代表736个唯一问题。

## 完整模型对照结果

| 模式 | prefix | MTP | 基线两新进程自重复 | PR两新进程自重复 | 跨源码r0 | 跨源码r1 |
| --- | --- | --- | --- | --- | --- | --- |
| graph | off | 0 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| graph | on | 0 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| eager | off | 0 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| eager | on | 0 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| graph | off | 3 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| graph | on | 3 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| eager | off | 3 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |
| eager | on | 3 | 逐位一致 | 逐位一致 | 逐位一致 | 逐位一致 |

每个“逐位一致”覆盖对应完整一轮的1408 token和7040分数，以及完整top-k token集合；输入长度、生成长度、cache命中计数、finish reason、配置与有效cache配置同时核对。没有把两个已经不同的生成历史下的后续分数混在一起比较。

另外，与原未修改默认路径逐一比对每轮前21个短/批/混批响应：32/32轮全部保持原有token与score bits。原长上下文无序路径自身不重复，因此没有要求新控制匹配其中任意一次随机顺序结果。

## 新请求重放：跨cache状态的数值现象记录

这里的top-k指**API返回的生成候选top-5及其logprob**，不是对sparse_index_topk内部注意力KV索引集合作了新的差异判定。记录的对照关系是：同一版本miss→miss可重复、hit→hit可重复；迁移前后miss→miss和hit→hit分别逐位对齐；同一版本miss→hit则允许记录数值差异，不将缓存状态不同的两次执行混为同条件迁移比较。

每轮额外核对单请求重放、8请求批次重放、长请求重放，共320个响应配对：

| 检查 | 结果 |
| --- | --- |
| 完整生成token序列 | 320/320相同 |
| 完整top-k集合/score bits | 304/320相同 |
| 相同cache命中计数下的差异 | 0 |
| 跨cache命中状态的数值变化记录 | 16，均为prefix开启的长请求首次miss→hit |

这16对的cached_tokens从0变为32576；均在首个生成位置就出现top-k集合/分数差异，生成token序列仍全部相同。两源码、eager/graph和非MTP/MTP3都出现同类现象。首差位置、此前一致的生成历史、候选ID和分数位模式已保存在审计文件中。

该比较改变了cache命中状态及prefill工作量，属于不同执行路径下的浮点数值比较。本文记录现象和配置，不进一步指认GEMM、NCCL或某条具体指令，也不将它列为待修复的迁移问题。原始审计中cross-hit比较的非相等计数保留，但不改变32项同条件主比较全部通过的结论。

原始JSON的different_hit_condition_failures字段沿用零容差比较器的命名，值16表示16个跨缓存状态的非相等配对，不代表16个迁移回归；原始字段、数据和归档SHA均保持不变。

### 代表性配置和原始结果

取基线侧graph、prefix on、非MTP的第一个新进程（文件glm-det-full-20261117-graph-p1-m0-base-r0.json）。配置为完整GLM-5.3 FP8 checkpoint、BF16激活、TP8、block64、4276 GPU blocks、max batch8、prefill上限2048、greedy/seed42；选择器deterministic=True。对应PR侧和双方第二次新进程在各自相同缓存状态下的结果均相同。

| 项目 | 首次cache miss | 随后cache hit |
| --- | --- | --- |
| 完整prompt token数 | 32603 | 32603 |
| cached_tokens | 0 | 32576 |
| 未缓存输入token数（input−cached） | 32603 | 27 |
| 生成token数 | 32 | 32 |
| 首个生成token ID | 785 | 785 |
| 完整32-token生成序列 | 两次相同 | 两次相同 |

首个生成位置的原始返回值如下，数值按保存值列出，未通过舍入展示制造一致：

| 候选位置 | miss token ID | miss logprob | hit token ID | hit logprob |
| --- | --- | --- | --- | --- |
| 1 | 785 | -0.0002803802490234375 | 785 | -0.00006198883056640625 |
| 2 | 154842 | -8.25 | 154842 | -9.875 |
| 3 | 10056 | -11.75 | 10056 | -13.25 |
| 4 | 1654 | -13.3125 | 1654 | -13.375 |
| 5 | 87499 | -14.5625 | 40 | -14.3125 |

此示例全部32个生成位置的score bits存在跨miss/hit差异，9个位置的top-5候选集合不同，但生成token全部相同。图模式与eager下对应长请求事件的记录相同。完整配置、每项二进制表示和32轮原始结果见本文末尾的证据归档与审计链接。

## 性能和交付边界

2026-11-18已补充五对AB/BA的短请求和32K保序迁移性能对照，并将同一基线保序开/关的独立开销另列。[最新性能、范围与原始证据](cuda_warm_performance_20261118.md)。初次正确性运行的加载/编译耗时仅属历史执行记录，不用于推理性能结论。

保序控制没有写入PR #5134，Intern-S2的BF16通信/FP32累加/固定rank顺序方案也没有在本轮实现。后续生产策略接入、一般batch不变性及当前成本实验未覆盖的性能范围仍见[独立任务Prompt](cuda_determinism_independent_task_prompt_20261117.md)；上述已记录的GLM跨prefix miss→hit数值现象不再作为本次排查或新增修复要求。

用户断网期间，GPU主机的独立后台守护持续观测任务；本轮原控制器正常完成，没有重启续跑或覆盖失败。任务结束时8卡恢复初始空闲占用（GPU0约116MiB，其他约4MiB）。

## 证据与复现

本机根目录仍为`/home/daixu/data/pr5134-cuda-20261116`，新增目录`glm-stable-full-20261117`。保留原32个完整输出JSON、8个配置比较、每轮命令/环境/源码SHA、各进程导入审计、原短请求对照、跨cache重放首差、依赖与checkpoint核对、源patch、控制器/后台守护日志及脚本。

- [最终汇总与核对](evidence/glm53-stable-fullmodel-20261117/summary.json)
- [完整配对与重放审计](evidence/glm53-stable-fullmodel-20261117/replay-and-primary-audit.json)
- [固定源码及计划](evidence/glm53-stable-fullmodel-20261117/source-and-plan.json)
- [checkpoint核对](evidence/glm53-stable-fullmodel-20261117/checkpoint-recheck.json)、[依赖核对](evidence/glm53-stable-fullmodel-20261117/dependency-recheck.json)
- [原始证据归档](evidence/glm53-stable-fullmodel-20261117/evidence.tar.gz)、[逐文件SHA清单](evidence/glm53-stable-fullmodel-20261117/manifest.json)

归档包含1590个数据文件及files.json索引，共4757662 bytes，SHA256为`d51bd05c986e409717c147026f1a29a5774c75b9db66bcbe11cbb690159cb37b`；已从归档逐文件重新读取并验证SHA。权重没有重复上传，已有权重与原始默认失败记录全部保留。
