# gated norm性能复核：当前状态

更新：2026-11-18，Asia/Shanghai。

最新完整模型对照见[2026-11-18性能复核](a3_warm_performance_20261118.md)。三个norm矩阵各8配对已全部完成。Agents64-token C1流式速度和TPOT通过±5%等效；但C4吞吐−5.35%，95%区间[−7.82%,−2.92%]，C1 TTFT+6.25%，区间[+2.87%,+9.33%]，两项均有7/8配对呈不利方向，正在独立定位。

Agents256-token C1 TTFT+5.40%（区间[−2.63%,+7.91%]），C4吞吐+1.03%（区间[−1.85%,+10.52%]）；Sage64-token C1 TTFT+2.19%（区间[−9.32%,+8.41%]）。其余指标和不确定性见完整表，不能整体标通过。

## 比较范围

本项为已原生迁移的342db85e→5f1b1b3c，两个版本均不使用dlinfer。它不能代替旧dlinfer→native的HTTP验收；通过和未验收项以[最新主表](a3_warm_performance_20261118.md)为准。

## PR 与 CI 记录

PR #5134当前代码2ccf38a4，既有14/14门禁通过；CI结果不替代性能验收。[CI完整记录](ci_docker_recovery_20261116.md)。

## 历史诊断

旧同worker替换、NPU二进制、设备计时、GC及主机等待定位已整体移入[old归档](old/gated_norm_performance_20261116.md)。原始失败和日志均保留；CPU18具体回调及永久修复仍见[独立主机报告](ray_host_latency_20261117.md)。
