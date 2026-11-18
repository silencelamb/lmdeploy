# A3 配对结果与统计复核

[对照报告](../../a3_c4_c8_performance_20261118.md)覆盖 Agents-A1（非 MTP）和 Sage-A1.5（MTP3），两者均为单实例 TP2、BF16、CANN 9.1.0。迁移基线为 LMDeploy `fb5de87a45df9999a19a569ed9441df8edfa6975` 与 dlinfer `3b9096341a41e9d067104a39d914e47ca2d7ec50`，原生版本为 `2ccf38a43f0fdc7415ce4ce9f177852bdacc9966`。两侧使用相同依赖栈和 checkpoint，原生环境检查覆盖 driver 和两个 worker 的 dlinfer 导入边界。

- [完整指标 CSV](results/comparison.csv)含绝对值、单位、配对变化、区间与判定。[配对数据 JSON](results/comparison.json)记录各自的八次旧版和新版读数、每对变化、形状覆盖及输出核对结果。
- [复测对照 CSV](results/repeat-comparison.csv)保留原始方向和独立确认阶段结果。[首轮 HTTP 和离线对照](results/first-http-offline.csv)给出之前的 C1 与 C4 数据。
- [测试判据](ACCEPTANCE.md)给出请求、预热和准入条件。[固定八行复测](results/packed8-matched-comparison.json)另用 99-token 提示，每模型两对。[测量次序核对](inventory/packed8-order-confirmation.md)说明这些数据的使用范围。

## 复算 HTTP 吞吐

在本目录运行下列 Python 代码，需要 NumPy。每个数字来自独立启动的进程对。每对先算新版相对旧版的吞吐变化，再取八个变化的中位数。有放回重抽进程对 10,000 次后，取这些中位数的中央 95% 区间。吞吐越高越好，TTFT 与 TPOT 越低越好。

```python
import json
from pathlib import Path

import numpy as np

data = json.loads(Path("results/comparison.json").read_text())
samples = np.random.default_rng(20261118).integers(0, 8, size=(10000, 8))
for model in ("http-agents", "http-sage"):
    for concurrency in ("1", "4", "8"):
        metric = data[model]["levels"][concurrency]["metrics"]["throughput"]
        delta = 100 * (np.array(metric["new_blocks"]) / metric["old_blocks"] - 1)
        center = np.median(delta)
        interval = np.quantile(np.median(delta[samples], axis=1), [0.025, 0.975])
        np.testing.assert_allclose(center, metric["paired_median_pct"])
        np.testing.assert_allclose(interval, metric["ci95"])
        print(model, concurrency, round(center, 2), interval.round(2))
```
