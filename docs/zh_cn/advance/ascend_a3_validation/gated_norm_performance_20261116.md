# gated norm性能复核：当前状态

更新：2026-11-18，Asia/Shanghai。

最新完整模型对照见[2026-11-18性能复核](a3_warm_performance_20261118.md)。Agents256-token的8个独立进程配对已完成：C1 TTFT +5.40%，95%区间[−2.63%,+7.91%]；C4吞吐+1.03%，区间[−1.85%,+10.52%]。仍未证明严格±5%等效，不能整体标通过。Sage与Agents64-token场景继续运行。

## 比较范围

本项为已原生迁移的342db85e→5f1b1b3c，两个版本均不使用dlinfer。它不能代替旧dlinfer→native的HTTP验收；通过和未验收项以[最新主表](a3_warm_performance_20261118.md)为准。

## PR 与 CI 记录

PR #5134当前代码2ccf38a4，既有14/14门禁通过；CI结果不替代性能验收。[CI完整记录](ci_docker_recovery_20261116.md)。

## 历史诊断

旧同worker替换、NPU二进制、设备计时、GC及主机等待定位已整体移入[old归档](old/gated_norm_performance_20261116.md)。原始失败和日志均保留；CPU18具体回调及永久修复仍见[独立主机报告](ray_host_latency_20261117.md)。
