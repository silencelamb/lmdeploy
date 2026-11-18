# A3 充分预热性能证据

六个正式矩阵的归档名以实验 label 和 complete-v1 结尾。另两个 complete-v1 包记录分层 profile。supplement-r2 包含 pilot、Sage draft 覆盖和 norm 同 worker 的失败及成功记录。prefill-crossover-r0 包含共享 decode 图的两组交叉实验。每个压缩包的 manifest 列出包和各成员的 SHA256、字节数及本机完整服务日志的哈希。归档写入后已经逐成员读回校验。

在 A3 实验根目录内解包，保留 results、logs 和 artifacts 的相对路径。各 logs/*.json 含完整命令、工作目录、版本、脚本哈希、亲和性和退出状态；artifacts/*-scripts 含对应脚本快照与导入 guard。复现模型实验使用固定 checkout、Python 环境、模型和新的 label。将本目录额外提供的分析程序放在实验根目录的 scripts 下，聚合 JSON 输出到 results。HTTP 窗口分析需要相同模型的 tokenizer.json 和 tokenizers 库，不加载模型权重。

```bash
cd /root/workspace/a3-perf-warm-20261118
python scripts/analyze_norm_components_r2.py
python scripts/norm_pipeline_boundaries.py
python scripts/norm_c4_pair_stages.py
envs/native-legacy/bin/python scripts/http_driver_sse_windows.py
envs/native-legacy/bin/python scripts/http_window_tail_event.py
python scripts/analyze_crossover.py agents-norm-crossover-r0
python scripts/analyze_gdn_crossover.py
```

分析时将 rank 视为同一次 TP2 运行的相关观测。组件的墙钟时间包含 host enqueue 和等待，更深的子调用嵌套于父调用中。正式验收定义和已通过项目见[主报告](../../a3_warm_performance_20261118.md)。
