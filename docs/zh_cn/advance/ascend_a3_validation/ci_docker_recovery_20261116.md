# PR #5134：Jetson依赖修复与CI完整回归

本轮按用户要求持续处理可完成的CI。迁移源码基线为`f64ab506`，当前验收提交为
`2ccf38a43f0fdc7415ce4ce9f177852bdacc9966`。本文按实际完成结果更新；
不能把进行中、取消或旧HEAD的通过状态计为本轮通过。

## 问题与归因

上一轮完整A100单测已达到6427通过、400跳过、0失败，ETE、lint及A3镜像也通过。
CUDA12.8/Python3.13曾因hosted runner失联失败，单独复跑后通过；原始日志没有
OOM、内核或网络故障的直接证据，所以底层触发原因仍不应写成已确认的OOM。
另五个CUDA配置被矩阵的默认fail-fast策略取消，它们没有完成验证。

Jetson是独立的继承依赖问题。旧Docker配方安装未固定版本的`cudss`元包；官方APT
仓库现将它解析到cuDSS1，而Jetson AI Lab的Torch2.11.0 wheel需要`libcudss.so.0`。
此错误发生在Torch导入，不能归为Ascend算子精度或MTP算法回归。

## 修复及失败迭代

1. `55a08ee6`改用官方`libcudss0-cuda-12=0.8.0.10-1`运行库，显式选择SONAME0。
2. `b547ca0b`设置CUDA矩阵`strategy.fail-fast:false`。八个CUDA/Python组合继续独立执行，
   失败仍报告失败，不增加continue-on-error或删减检查。
3. 首版Jetson镜像已成功构建并安装0.8.0.10，但`lmdeploy check_env`仍报找不到SONAME0。
   检查真实Deb包后发现，库位于`/usr/lib/aarch64-linux-gnu/libcudss/12`，runtime包没有
   ldconfig配置或触发器。`2ccf38a4`将该目录写入镜像的`ld.so.conf.d/cudss.conf`，再执行
   `ldconfig`，使动态链接器能够找到真实的SONAME0库。

这里修复的是版本选择和库搜索路径，没有将不兼容的SONAME1库冒充为SONAME0。
三项改动以独立CI提交记录；LMDeploy推理代码、CUDA源码及Python requirements与f64
相同，CANN保持9.1.0。此前模型/性能结果仍归属于实际测量版本，不改写为新的测量。

## 本地验证与证据

[完整证据索引](evidence/ci-jetson-runtime-20261116/index.json)记录原始日志、命令、失败、包哈希与对照。

- 官方包SHA256为`f66d3099cbd962c6f6b135415e562d98cb5212c50a5d6286f0b860d10ecb50ac`，
  完整下载后与官方APT元数据相符；实际ELF的SONAME为`libcudss.so.0`。
- 在仅解包到独立目录的真实库上运行ldconfig对照：注册前缓存中没有cuDSS，注册后
  找到`libcudss.so.0`和threading layer；宿主机`/etc/ld.so.cache`哈希保持不变。
- 两轮适用pre-commit检查通过，最终提交diff的SHA与检查过的工作树一致。
- 首次整包下载发生IncompleteRead，随后通过保存分片、断点续取和完整SHA验证恢复。
  第一版镜像失败、下载失败及每个分片的重试记录均保留，不能改写为首试通过。

官方cuDSS0.8安装文档将SONAME0和CUDA12.x列为对应配置；静态库查找验证只覆盖
动态加载路径，真实Torch导入仍以Jetson镜像CI为准。没有Jetson设备，不宣称实机推理
或稀疏求解数值已经验证。

## 远端CI结果

截至2026-11-16 23:55 Asia/Shanghai的当前HEAD结果如下。仍在执行的检查未计为通过，持续跟进目标保持进行中。

| 检查 | 当前状态 | 原始任务 |
| --- | --- | --- |
| test_docker_image (cu130, 3.13) | 通过 | [job133371113408](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113408) |
| test_docker_image (cu128, 3.10) | 通过 | [job133371113390](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113390) |
| test_docker_image (cu130, 3.12) | 执行中 | [job133371113275](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113275) |
| test_docker_image (cu128, 3.13) | 执行中 | [job133371113268](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113268) |
| test_docker_image (cu130, 3.11) | 执行中 | [job133371113137](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113137) |
| test_docker_image (cu130, 3.10) | 通过 | [job133371113106](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113106) |
| test_docker_image (cu128, 3.12) | 通过 | [job133371113034](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371113034) |
| test_docker_image (cu128, 3.11) | 执行中 | [job133371112983](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371112983) |
| test_jetson_docker_image | 通过 | [job133371112793](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371112793) |
| test_ascend_docker_image | 通过 | [job133371112386](https://github.com/InternLM/lmdeploy/actions/runs/43887914029/job/133371112386) |
| unit_test | 通过 | [job133371057262](https://github.com/InternLM/lmdeploy/actions/runs/43887913961/job/133371057262) |
| pr_functions_test | 执行中 | [job133370945520](https://github.com/InternLM/lmdeploy/actions/runs/43887914323/job/133370945520) |
| lint | 通过 | [job133370934220](https://github.com/InternLM/lmdeploy/actions/runs/43887914110/job/133370934220) |
矩阵为cu128/cu130与Python3.10/3.11/3.12/3.13的八种组合，另有Jetson、A3、lint、
完整unit及ETE，共十三项检查。被新提交替换的旧轮取消记录与最终轮结果分开保存。


### 当前单测计数与合并预览

当前完整单测为**7452通过、400跳过、1903条warning**。本轮CI实际checkout的是
`fea0d6fd9975d1a23d260010eef4b10d95ed5571`，父提交为当前上游
`95d2d6280ed4946fd498d6026f338c7bd41bfb63`和PR HEAD `2ccf38a4`。
上轮6427通过的CI checkout为`0e15234989953acd0fa9dbae4573545ab2b6c6cf`，父提交是
`442097e8`和`f64ab506`。新增的1025个用例来自期间合入上游的Anthropic、XML/schema
和GLM聊天模板测试；它们不应算作这三个CI修复提交新增的测试。硬件、Python3.12.13、
Torch2.10.0+cu128和TorchVision0.25.0+cu128保持一致。

固定版本的A3迁移比较与这个GitHub合并预览测试是不同的验证记录，不能把旧模型结果
重标为新上游版本上的测量。单测400项skip没有计入通过数。

Jetson当前已经实际通过Build Docker image、`lmdeploy check_env`和Dive三步。
这确认了版本选择与加载路径修复；实机GPU推理仍未在本轮覆盖。

## 来源

- NVIDIA官方APT仓库：https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/
- cuDSS0.8安装说明：https://docs.nvidia.com/cuda/cudss/0.8.0/getting_started.html
- 当前PR：https://github.com/InternLM/lmdeploy/pull/5134
- 研究分支：https://github.com/silencelamb/lmdeploy/tree/research/ci-jetson-runtime-20261116
