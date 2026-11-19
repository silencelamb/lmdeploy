# Agents-A1 的 HTTP 流式输出间隔

迁移前的基线和当前 PR 都出现过同类停顿：服务端连续输出两个 token，本机 HTTP 客户端却相隔半秒多才读到这两个 token。已有记录表明这一现象在迁移前就存在。样本不足以判断迁移是否改变了它的发生频率。

## 历史 C4 批次中的 12–14 ms 和 528–530 ms

历史原生版本 `6421cefe` 的一个四并发批次（`b01`，`rep6`）中，四个请求都在第 118 到第 119 个 token 出现了这个间隔。对每个请求，在服务端用 `AsyncEngine.generate` 输出第 119 个 token 的时刻减去输出第 118 个的时刻，得到 **12.06–13.77 ms**。在客户端，对同两个 token 的非空 SSE 消息读取时刻做同样的减法，得到 **527.98–529.77 ms**。两个范围都取这四个请求的最小值到最大值。例如同一条请求，服务端间隔 13.77 ms，客户端间隔 529.77 ms。

配对旧版在相同位置的客户端间隔为 25.01–25.10 ms。主机 CPU 采样在该批次之后才启动，缺少将这次交付停顿归到具体线程或定时器的同期记录。[历史批次新旧两侧的逐 token 记录](https://github.com/silencelamb/lmdeploy/blob/0bab7fe0172db1180c7acc6c9824f1b0dd9a03b4/docs/zh_cn/advance/ascend_a3_validation/evidence/a3-warm-attribution-20261118/http-largest-window-residual-r0.json) · [原始测量与分析程序](https://github.com/silencelamb/lmdeploy/tree/0bab7fe0172db1180c7acc6c9824f1b0dd9a03b4/docs/zh_cn/advance/ascend_a3_validation/evidence/a3-warm-attribution-20261118)

## 基线与当前 PR 的独立复测记录

从 8 对新旧服务进程的正式测量中，按提示匹配请求，逐 token 核对服务端输出与客户端收到的文本。复核相邻两个 token 各自单独输出的间隔，筛选条件为客户端间隔 ≥500 ms 且同两个 token 的服务端间隔 ≤50 ms。下表分母为各版本的实测批数，同一批多个请求命中只计一批。基线为 LMDeploy `fb5de87a` + dlinfer `3b909634`，当前 PR 为 LMDeploy `2ccf38a4`。[独立复测配置](../../a3_c4_c8_performance_20261118.md)

| 并发 | 基线命中批数 / 实测批数 | 当前 PR 命中批数 / 实测批数 | 命中请求的服务端间隔 | 命中请求的客户端间隔 |
|---|---:|---:|---:|---:|
| C1 | 1 / 200 | 0 / 200 | 12.03 ms | 554.39 ms |
| C4 | 0 / 128 | 1 / 128 | 15.92–16.87 ms | 568.02–568.99 ms |
| C8 | 1 / 64 | 0 / 64 | 15.21–18.13 ms | 564.19–577.59 ms |

[分版本批次统计](confirmation-batch-counts.csv)列出各并发的命中批数和请求数。[逐请求间隔](verified-half-second-gaps.csv)列出历史案例和独立复测中的全部命中间隔。表中的 `next_token` 从 1 开始编号，例如 119 对应第 118→119 个 token。`server_ms` 和 `client_ms` 是两端各自测得的间隔，`excess_ms` 是后者减去前者。
