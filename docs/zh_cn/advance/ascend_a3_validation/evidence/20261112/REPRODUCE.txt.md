# REPRODUCE.txt

原始验证文件快照；源码 SHA256：`7871ff63b0e1f10d6fcdd270ecd1a105d533f24c4d4f28a603135790c117e869`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```text
本目录为 A3 实机验收的独立工作区。原 /root/workspace/{lmdeploy,dlinfer}
以及同事 /root/workspace/code 中的工作区文件未修改。

固定源码版本见 inventory/comparison-revisions.json：
baseline LMDeploy fb5de87a45df9999a19a569ed9441df8edfa6975
baseline dlinfer  3b9096341a41e9d067104a39d914e47ca2d7ec50
original native  39dbaa841080d9b9b6b0326197b9de7fe9495a1b
repaired native  6421cefea94edd350e419f556158639556357a27
clean dlinfer    05ede38ae879347be33c8f6ebb136d84f3f4ee88
两个 main 与迁移父提交一致。修复仅存在本地分支
codex/ascend-a3-validation-20261112，没有推送。

复用已保存环境，先 cd /root/workspace/a3-validation-20261112。
每次使用新的 --label，避免覆盖旧日志。示例：

python scripts/run.py --env baseline --label replay-baseline-eager -- \
  python scripts/pipeline_case.py --eager --scores --suite trace --new-tokens 64 \
  --output results/replay-baseline-eager.json
python scripts/run.py --env native --label replay-native-eager -- \
  python scripts/pipeline_case.py --eager --scores --suite trace --new-tokens 64 \
  --output results/replay-native-eager.json
python scripts/compare.py results/replay-baseline-eager.json \
  results/replay-native-eager.json --output results/replay-eager-comparison.json

MTP 重复性：在上述两个 pipeline 命令中移除 --eager，添加 --mtp 3 --prefix；
run.py 添加 --global-seed 42，使用不同 label/output 重复两次。
这项严格精度验收当前未通过，不能只凭退出码 0 判断成功。

接近上下文边界（两边都用同一组空闲设备，本次为逻辑 2,3）：
python scripts/run.py --env native --label replay-native-context --devices 2,3 -- \
  python scripts/pipeline_case.py --mtp 3 --prefix --no-thinking --respect-eos \
  --scores --suite long --length 262000 --new-tokens 64 \
  --output results/replay-native-context.json
同参数 --env baseline 生成参照后执行 compare.py。
实际输入长度必须从结果 input_token_len 读取，不能用 --length 代替实测。

HTTP（使用同一设备 0,1，前一个服务完全退出后再运行另一环境）：
python scripts/run.py --env baseline --label replay-baseline-http -- \
  python scripts/http_case.py --repeats 5 --output results/replay-baseline-http.json
python scripts/run.py --env native --label replay-native-http -- \
  python scripts/http_case.py --repeats 5 --output results/replay-native-http.json
python scripts/http_compare.py results/replay-baseline-http.json \
  results/replay-native-http.json --output results/replay-http-comparison.json
本地 127.0.0.1:25485；HTTP 客户端每请求新连接，不自动重试。
正式首组使用 5 轮；复测使用 7 轮并反转为 native -> baseline。
scripts/run_http_recheck.py 保存了该顺序和 npu-smi 采样方式；已有运行不能
直接覆盖，再跑前需换一组 label/output。三档并发为 1、4、16，每档独立预热。

MTP3 前缀缓存/精确答案专项（输入评分共同失败已另存，不算作通过）：
python scripts/run.py --env native --label replay-native-prefix-quality -- \
  python scripts/pipeline_case.py --mtp 3 --prefix --scores --suite extra \
  --skip-input-scoring --length 8192 --output results/replay-native-prefix-quality.json
同参数 --env baseline 后使用 compare.py。去掉 --skip-input-scoring 可重现
本次两边共有的 MTP max_new_tokens=0 输入评分崩溃。

独立 dlinfer MoE（该环境没有安装 LMDeploy）：
python scripts/run.py --env standalone --label replay-standalone-moe --devices 2 -- \
  python scripts/standalone_moe.py
其余算子 pytest 的完整命令在 logs/*operator-tests*.command.json。
standalone_moe.py 的结果文件名固定；需要保留重跑前的该结果副本。

环境创建依据：
- scripts/setup_envs.py 将当前镜像第三方 site-packages 复制成固定副本，
  排除 LMDeploy/dlinfer 的包、metadata、editable finder 和 .pth。
- 三个 venv 均不继承 system-site-packages；第三方依赖链接到固定副本，
  项目分别 editable 安装。没有升级或卸载镜像原环境的包。
- 对照完整安装命令为 logs/install-*.command.json；均 --no-deps
  --no-build-isolation。两个 dlinfer 安装增加 --ignore-requires-python。
- 原镜像 Python 3.12 / torch 2.10 不满足部分依赖声明；pip check 的
  实际非零结果在 inventory/*-environment.json，不能视为通过。
- native sitecustomize 使用 scripts/sitecustomize_native.py 的 import
  guard；追加 scripts/seed_startup.py 仅在 VALIDATION_GLOBAL_SEED 设置时
  固定每个解释器的随机种子。baseline/standalone 仅有这个可选 seed hook。
- triton-ascend-kernels 使用 sources/triton-ascend-kernels-snapshot，
  固定 commit 8e70a44afa537fcbcdc6d59e6d2196a38f1e7b6f 加镜像已有补丁；
  补丁见 inventory/code-triton-ascend-kernels.json，源码文件哈希另存。
- CANN/ATB/驱动使用当前容器及宿主提供的同一份安装，版本和路径见 inventory。
- 模型始终是同一份只读 /afs-weights 权重。完整文件 SHA256、大小及 mtime
  在 inventory/model-*.json；没有下载或修改权重。

日志、退出码、开始/结束时间、命令、运行环境差异在 logs/*.command.json。
较后运行还保存了每次 harness 脚本副本及其 SHA256、源码 HEAD 与 diff，
位于 artifacts/harness-snapshots/<label>。早期启动/测试命令亦保留了完整日志。
原始 token/logprob/请求/时间数据在 results；跳过和失败保留在 JUnit XML。
ACCEPTANCE.txt 是比较前固定的判据，未为 MTP 差异放宽阈值。

工作区初始化脚本是审计资料；重新创建到其他路径时，应按这些已记录的
commit 新建 worktree/venv，并重新设置 ROOT 及 .pth，不能直接复用旧日志 label。
不要执行同事 install.sh 中的全局 uninstall、rm -r，也不要修改其源码目录。

用户追加的 Agents-A1：
使用同一个 run.py，把 pipeline_case.py 的 --model 改为 /afs-weights/Agents-A1。
按 Agents-A1 部署文档不传 --mtp，默认 MTP 关闭。实际执行矩阵在
scripts/run_agents_cases.py；HTTP 使用 http_case.py --model /afs-weights/Agents-A1
--model-name agents-a1 --mtp 0。精度、上下文边界与性能的完整命令都已记录。

Mooncake 扩展环境（与前面的三个环境分开）：
baseline-mooncake / native-mooncake 已安装 NPU wheel 0.3.13.post1。
wheel 位于 artifacts/wheels，其 SHA256 在 inventory/mooncake-wheel.json；
安装 argv 和完整日志为 logs/install-*-mooncake-0.3.13.post1.*。
额外 worktree 的 commit 与分支见 inventory/mooncake-comparison-revisions.json。
二者均额外允许 Store protocol=ascend，属于共同平台接入补丁，不能当作
未经修改的 main；补丁说明见 artifacts/mooncake-common-adapter.txt。

底层双 NPU 直接缓冲区探针（先确认逻辑设备2,3及本地55051/55052空闲）：
python scripts/run.py --env native-mooncake --label replay-mooncake-sdk \
  --devices 2,3 --timeout 240 -- env ASCEND_ENABLE_USE_FABRIC_MEM=1 \
  python scripts/mooncake_sdk_probe.py --output results/replay-mooncake-sdk.json
脚本只启动自己管理的本地 master，并在 finally 退出它和两个探针进程。
此项当前失败，不能将完整数据传输视为已验证。0.3.13.post1 中环境变量
只要存在就会启用 fabric，值0也一样；对照须使用 env -u ASCEND_ENABLE_USE_FABRIC_MEM。

模型启用 Store 的模板为 scripts/run_mooncake_models.py 和 configs/mooncake-a3.json，
使用 MooncakeStoreConnector / kv_both，master 127.0.0.1:55051，Ascend transport，
每 worker global segment2GB、local buffer256MB。Sage MTP3，Agents MTP关闭。
无需连接 /afs-weights/mooncake.json 中的外部 master；本次从未使用该远端。
模板带已有测试队列的完成等待条件和固定 label，重跑时须调整这些条件/label。

Agents-A1固定批次引擎内对照（同HTTP请求文本及输入长度）：
python scripts/run.py --env native --label replay-agents-pipeline-c4 -- \
  python scripts/pipeline_case.py --model /afs-weights/Agents-A1 --prefix \
  --new-tokens 128 --suite bench --bench-concurrency 4 --bench-http-workload \
  --no-thinking --repeats 7 --output results/replay-agents-pipeline-c4.json
同参数baseline后用compare.py。该bench模式不采分数，主要输出bench指标；
不能把无score数据的numerical_status当成精度通过。

evidence.tar.gz 是便于转移的日志/结果/脚本/说明/补丁/wheel/外部kernel快照包，
其中每个文件的SHA256在EVIDENCE-SHA256.json，压缩包摘要在evidence.tar.gz.sha256。
完整LMDeploy/dlinfer工作区和五个venv仍保留在本机，未打包进该证据归档。

```
