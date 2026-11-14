# A3 验证复现入口

这些测试已经在指定 A3 镜像执行；下列命令用于复用保存的环境，不是任意机器上的一键部署脚本。新机器需先准备相同模型、运行时与隔离环境。完整源码版本见 [总览](README.md)。

## 环境与数据

原镜像为 `registry2.d.pjlab.org.cn/ccr-deeplink/lmdeploy_dlinfer:0.18.0-20261102T080655Z`。实际为 aarch64、8 个 64 GiB NPU chip，driver 25.5.6、CANN 9.1.0、Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4、Transformers 5.14.1。实际软件清单和模型文件哈希见 `evidence/20261112/inventory/`。

旧基线、新原生、dlinfer standalone 的 venv 相互隔离；native 没有 dlinfer，standalone 没有 LMDeploy。不得将另一个环境的项目源码加入 PYTHONPATH。当前镜像的部分依赖声明不满足，pip check 非零；复现时应保留这一限制，不能把重新搭建不同依赖的结果当成同一次比较。

模型为同一份 `/afs-weights/Sage-A1.5`、`/afs-weights/Agents-A1`、`/afs-weights/Qwen3-8B`，没有随报告上传权重。外部 triton-ascend-kernels 使用固定 commit 加机器原有补丁的快照；只有远端 commit 而没有该补丁时，尚未得到完全相同的环境。

## 已保存环境中重跑

每次使用新的 label/output。先确认所用逻辑设备空闲，按顺序执行 baseline 和 native；不要并行跑性能对比。

### 非 MTP 数值与 MTP 重复性

```bash
cd /root/workspace/a3-validation-20261112
python scripts/run.py --env baseline --label replay-20261114-baseline-eager -- \
  python scripts/pipeline_case.py --eager --scores --suite trace --new-tokens 64 \
  --output results/replay-20261114-baseline-eager.json
python scripts/run.py --env native --label replay-20261114-native-eager -- \
  python scripts/pipeline_case.py --eager --scores --suite trace --new-tokens 64 \
  --output results/replay-20261114-native-eager.json
python scripts/compare.py results/replay-20261114-baseline-eager.json \
  results/replay-20261114-native-eager.json \
  --output results/replay-20261114-eager-comparison.json
```

MTP 重复性控制：移除 `--eager`，增加 `--mtp 3 --prefix`；在 run.py 参数中增加 `--global-seed 42`，每个环境独立进程重复两次，分别检查同版本和跨版本差异。Agents 文档配置关闭 MTP，模型参数使用 `--model /afs-weights/Agents-A1`。

### HTTP C4 与固定批次对照

```bash
python scripts/run.py --env baseline --label replay-20261114-baseline-http -- \
  python scripts/http_case.py --repeats 7 \
  --output results/replay-20261114-baseline-http.json
python scripts/run.py --env native --label replay-20261114-native-http -- \
  python scripts/http_case.py --repeats 7 \
  --output results/replay-20261114-native-http.json
python scripts/http_compare.py results/replay-20261114-baseline-http.json \
  results/replay-20261114-native-http.json \
  --output results/replay-20261114-http-comparison.json
```

此模板默认覆盖 C1/C4/C16，端口 `127.0.0.1:25485`。Agents 增加 `--model /afs-weights/Agents-A1 --model-name agents-a1 --mtp 0`。性能复核应再次反转环境顺序。总吞吐含 prefill/HTTP；流式输出速度由 SSE 首尾时间估算，两者不是同一个指标。

固定 C4 pipeline 对照使用 `pipeline_case.py --model /afs-weights/Agents-A1 --prefix --new-tokens 128 --suite bench --bench-concurrency 4 --bench-http-workload --no-thinking --repeats 7`，两环境参数相同；该 bench 不采集分数，不按数值验收通过计。

### Mooncake fabric 和外部缓存

```bash
cd /root/workspace/a3-mooncake-pr5030-20261113
python scripts/run.py --env native --label replay-20261114-native-fabric --devices 2,3 \
  --timeout 240 -- env ASCEND_ENABLE_USE_FABRIC_MEM=1 \
  python scripts/fabric_probe.py --segment-gib 1 --io-api batch \
  --output results/replay-20261114-native-fabric.json

python scripts/run.py --env baseline --label replay-20261114-baseline-agents --devices 0,1,2,3 \
  --timeout 1500 -- env ASCEND_ENABLE_USE_FABRIC_MEM=1 \
  python scripts/model_external_cache.py --model /afs-weights/Agents-A1 --mtp 0 \
  --label replay-20261114-baseline-agents \
  --output results/replay-20261114-baseline-agents.json
```

再用 native 和新 label/output 执行相同模型命令。Sage 改为 `--model /afs-weights/Sage-A1.5 --mtp 3`。服务自管本地 55061/55062 端口；producer 占逻辑设备 0,1，fresh consumer 占 2,3。配置 `protocol=ascend`、`device_backend=ascend`、fabric global segment 8 GiB；PR 分支有效 local buffer 为 0。

```bash
python scripts/collect_runtime_logs.py
python scripts/analyze_external_cache.py results/replay-20261114-baseline-agents.json \
  results/replay-20261114-native-agents.json \
  --output results/replay-20261114-agents-comparison.json
```

上述验证必须使用独立 Mooncake 集成源码，迁移主分支没有 PR #5030 的适配。跨实例外部命中不能用本地 prefix hit 替代。

### TileLang 和 worker 单测

两套 Mooncake 环境现已补齐 `tilelang==0.1.15`、`apache-tvm-ffi==0.1.12`、`torch-c-dlpack-ext==0.1.5`、`ml-dtypes==0.6.0`、`z3-solver==4.15.4.0`。版本、wheel URL/SHA256、变更链接、freeze 与审计在 `evidence/20261113/inventory/tilelang-followup/`。

```bash
python scripts/run.py --env baseline --label replay-20261114-baseline-worker --devices 4 \
  --timeout 300 -- python -m pytest -q \
  sources/lmdeploy-baseline/tests/pytorch/kv_connector/test_mooncake_store_worker.py \
  --junitxml=results/replay-20261114-baseline-worker.xml
python scripts/run.py --env native --label replay-20261114-native-worker --devices 5 \
  --timeout 300 -- python -m pytest -q \
  sources/lmdeploy-native/tests/pytorch/kv_connector/test_mooncake_store_worker.py \
  --junitxml=results/replay-20261114-native-worker.xml
```

这两个结果各 101/101。补依赖前的 100/101 及失败栈也保留；GLM/MTP 用例是 CPU 缓存布局往返测试，不能当成 GLM 整模型或 CUDA kernel 执行证明。新的复现使用补依赖后的环境；此前 NPU 模型结果对应安装前的环境清单。

## 从 GitHub 恢复原始 harness

为了便于 GitHub 阅读，原始脚本以 `*.py.md` 的 Python 代码块保存；没有格式化或改写脚本体。以下代码只提取文件并验证源码哈希，不运行脚本：

```python
import hashlib
import json
from pathlib import Path

docs = Path('docs/zh_cn/advance/ascend_a3_validation')
index = json.loads((docs / 'evidence/source-index.json').read_text())
output = Path('/tmp/ascend-a3-harness-restored')
for relative, info in index.items():
    if info['encoding'] != 'markdown-fence-python':
        continue
    published = docs / relative
    body = published.read_text().split('```python\n', 1)[1]
    source = body.rsplit('\n```\n', 1)[0].encode()
    assert hashlib.sha256(source).hexdigest() == info['source_sha256']
    day = Path(relative).parts[1]
    target = output / day / 'scripts' / published.name.removesuffix('.md')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
```

脚本包含原机路径、已用 label、旧队列等待条件和环境初始化假设。重新部署前应阅读脚本，并按固定 commit 重建 sources/venv、CANN 路径、外部 kernel 快照与只读模型路径；不要直接执行整个历史队列或同事安装脚本。Git 中的选定结果用于审计摘要；全量 token/logprob/SSE 和每次 harness 快照在本机完整归档中。

## 完整证据包

[local-archives.json](evidence/local-archives.json) 保存三份完整归档的路径、大小及 SHA256。报告分支不提交 wheel、权重或大归档；已选定文件的原始来源和哈希由 [source-index.json](evidence/source-index.json) 索引。阶段一、阶段二的原始复现说明和验收判据也以 Markdown 快照保留在各日期 evidence 目录。
