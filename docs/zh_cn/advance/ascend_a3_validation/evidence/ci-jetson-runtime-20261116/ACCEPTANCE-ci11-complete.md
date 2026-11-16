# CI持续跟进判据

补充完整性审计：除13项GitHub Actions check-runs外，必须检查commit statuses。
本轮发现额外的`docs/readthedocs.org:lmdeploy-zh-cn`失败状态，因此当前共14项门禁。
Read the Docs也必须实际构建成功，不能以13项Actions全绿代替完整CI通过。

用户要求持续处理PR #5134，能够完成的CI应实际通过。当前版本为
`b547ca0b89444dc6cb1db33b881e1d06aea60a4b`，上轮已验证版本为`f64ab506`。

- 以当前PR HEAD的GitHub结论为准：lint、完整unit、ETE、A3 Docker、Jetson Docker、8个CUDA/Python镜像组合，共13项检查均应实际完成并成功。取消、排队、进行中或被跳过的job不得计为成功。
- 单测自身已有的硬件条件skip如实报告；不能把这些skip当作硬件功能通过。保持原工作流测试命令、断言和矩阵成员。
- Jetson必须完成真实镜像构建、`lmdeploy check_env`和原有镜像检查。官方cuDSS包的版本、哈希及SONAME作补充证据；仅存在文件或通过静态分析不足以代替镜像导入验证。无Jetson设备，不宣称实机模型推理已验证。
- `fail-fast:false`只保证独立配置继续运行，失败仍导致CI失败；不使用continue-on-error、删除测试或伪造库名来制造通过。
- 新发现的实现回归直接修复并回归。继承的Jetson依赖漂移作为独立CI提交和独立归因记录，不能计为Ascend模型精度/性能回归。
- 运行时代码、CUDA源码和Python requirements与f64保持相同；本轮无需重跑全部A3迁移验收，不改变CANN9.1、NPU驱动、原生环境或Ray同步顺序。
- runner失联、外部下载失败与代码失败分别判断。合理重跑用于验证已经识别的临时故障；重复出现时继续收集具体证据，不无限重跑替代定位。
- 保存各次HEAD、job/run/attempt、命令、日志、失败和对照。重跑使用新label，报告维护在fork文档分支，并保留其他GPU验证记录。

当前两项修复分别为55a08ee6（Jetson cuDSS运行库）和b547ca0b（独立完成CUDA矩阵）。
本地适用pre-commit通过；完整远端CI结果仍需后续逐项收集。
