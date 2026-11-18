# 动态HTTP C4的独立性能复核

本目录不包含新增GPU运行。原始测量、token/score和日志位于[充分预热归档](../cuda-h200-warm-20261118/README.md)。解包raw归档后执行：

    python3 reperf_c4_performance_only.py --root /path/to/unpacked-raw --campaign /path/to/unpacked-raw/campaign --output /path/to/new-c4-audit

脚本只使用Python标准库。核对30个进程的配置、同一prompt token及顺序、600个响应的256输出token与length结束、零cache命中、资源和捕获开关；独立从请求耗时重算每批与五对统计，并要求与原数据完全相同。

[审计结果](performance-only-audit.json)保留原始源文件SHA256及各侧首次token差异。数值重复性仍为false；独立性能判据为true。它们是两个验收维度。
