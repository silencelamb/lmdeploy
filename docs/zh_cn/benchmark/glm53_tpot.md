# GLM-5.3-Flash 在 H200 上的生成执行对照

这里在相同 DeepGEMM MegaMoE 基础上改变 KDA prefill 实现，并单独测量已有服务在 MTP5 下的执行与混部表现。[MegaMoE 基线](glm53_sm90_megamoe.md) · [执行测量数据](../../../benchmark/glm53_mega_moe/results/h200_execution.json) · [FlashKDA 启用与复现](../../../benchmark/glm53_mega_moe/README.md#flashkda-prefill-on-h200)

MTP3 草稿词表分片、原调度同形状开关与 AgentX C90 四组集成结果见[草稿投影实测](glm53_mtp_draft_head.md)。

## FlashKDA prefill

GLM 的 45 层中有 34 层 KDA。单 H200、8,192 tokens、64 个 128 维 head 的递推（含 indexed state 读写）从 FLA 2.256 ms 降到 FlashKDA 0.640 ms。每条 64K prompt 分为 8 个 chunk，按单层耗时差估算可节约 0.440 秒；整模稳定段实际节约 0.423 秒。每 rank 处理 8 条 64K 输入的整组 prefill 从 47.22 降为 44.02 秒。MTP0 的 8K/256、C=64 滚动吞吐由 1,622 升至 1,702 tok/s（+4.92%）。每组两次启动、每次 256 条成功请求。

两种计算次序存在数值差异。64 条 64K 输入中 27 条首 token 相同，沿 FLA 轨迹的 512 个 decode 步有 31,927 / 32,768 个 argmax 匹配。16 条 64K 合成检索两组均答对。1,319 题 GSM8K 为 1,216 → 1,227 题，差 +0.83 个百分点，配对 95% 区间 [−0.30, +2.05]。FlashKDA 通过 `LMDEPLOY_GLM53_FLASHKDA_LIBRARY` 显式启用。

Weka 256K AgentX、C=32、MTP5、seed 42、同样的 36,430 个 KV 物理块，单组 1,800 秒。全机输出吞吐从 1,096.64 升至 1,127.82 tok/s（+2.84%），TTFT p50 从 692 降至 639 ms，TPOT p50 从 13.77 降至 12.91 ms。逐请求达标 1,796/2,304 → 1,970/2,366。两组预热分别有 177/183 次客户端未解析到可见内容，正式窗分别有 1/0 次。该组使用本地前缀缓存。

## MTP5 与同机 vLLM

8×H200、DP8 + EP8 + TP1、相同输入 ID、每 rank 可用 KV 容量约 90 万 token。LMDeploy 为 FlashKDA + MegaMoE，vLLM 为 `5328ae2fef` + MegaMoE。各做两轮。滚动负载为 8K/256、C=64，每轮 256 条。另用每 rank 4 条、合计 32 条热前缀请求，各生成 512 tokens。在干扰组，等全部探针至少输出 32 个 token，再投递 24 条未缓存的 8K prompt。

| 指标 | LMDeploy | vLLM |
| --- | ---: | ---: |
| C=32 单独生成，TPOT p50 | 4.41 ms | 4.66 ms |
| C=32 中途补入长 prompt，原请求 TPOT p50 | 8.74 ms | 7.64 ms |
| C=64 滚动负载，输出 tok/s | 1,121 | 1,536 |
| C=64 滚动负载，TPOT p50 / p95 | 47.14 / 88.71 ms | 33.22 / 62.14 ms |
| C=64 每轮平均可提交 token | 4.61 | 4.63 |

热前缀无补入的第二轮为 4.23 / 4.26 ms/token。补入后，在 LMDeploy 中得到完全相同的 512 个输出 ID 的 50 对探针，TPOT 配对差的中位数为 +3.82 ms。MTP5 在高并发混部的差距需要拆解每轮 draft、target 验证、状态推进和等待，并结合生成中用户的长停顿验收。上线优化主测 MTP3，MTP5 用于较宽验证对照。
