# 两台 A3 的 CPU 定时器停顿：工具、数据与复算

[返回调查报告](../../a3_timer_softirq_investigation_20261119.md)。A 是运行迁移对照的原机，B 是另外一台物理 A3。部分原始表用 `local` 指 A、`remote` 指 B。请求表里的 `old/new` 只表示 **A 上的软件版本**。两机时间线独立。公开时间戳使用每轮内的相对时间，保留原始精度。

## 采集工具

| 下载 | 环境、用途和输出 |
|---|---|
| [容器探针源码包](container-kit.tar.gz?raw=true) | 基础工具依赖 Linux、Python 3、C++17、make、zlib。含 `wakeup`（逐次唤醒）、`fine_irq`（睡眠或持续取时，独立核采软中断向量）、`probe.py`（全核扫描和摘要）。高级工具 `run_context.py` 同时记录自有线程切换、架构时钟 IRQ、时间片与系统总账，`analyze_context.py` 用 NumPy 离线对时。解压后有中文 README |
| [宿主机函数采集包](host-kit.tar.gz?raw=true) · [在线读主采集器源码](capture_timer.py) | 主工具 `capture_timer.py` 使用 Python 标准库和一个私有 tracefs 实例，采集指定核的 timer 回调和嵌套硬 IRQ。备用 `timer_capture.py` 使用系统级 perf。包内附执行命令、输出文件和判读方法。函数级在线采集需宿主内核开放事件；工作容器可离线解析 |
| [文件校验清单](SHA256SUMS) | 包含本目录的工具、数据、图片、复算程序和说明。保留完整目录时执行下方命令即可核对 |

只带回一个工具包时，可单独计算 `sha256sum *.tar.gz` 并与清单比对。两机原实验的可执行文件也核对过，`wakeup` 的 SHA256 为 `beaf5f02b154b9b637b36f426f078efba56f9fdafaa8ca3d0444ab20ab12af07`，`fine_irq` 为 `8ab4d37f64c51b60ad9a96e97c4ca52f57c0564f66251a52654f82e235519a9b`。源码重编译的二进制哈希还取决于编译器和构建选项。

## 实测数据

| 文件 | 内容、单位与分母 |
|---|---|
| [全核扫描](all-core-scan.csv) | 每机 319 行，每行一个 CPU，重复核在各波次的数据已合并。`seconds` 为该核实际测试时长，`over100/over300` 是超出 12 ms 期限的次数，`softirq_s` 是累计秒数 |
| [四核无轮询的全部唤醒](quiet-wakeups.csv.gz?raw=true) | 47,127 行。每行一个期限和实醒时刻，均为自该阶段启动以来的纳秒数。两者相减是额外等待。热点长窗起点间隔用连续长样本的应醒时刻之差 |
| [软中断向量窗口](vector-windows.csv) | 152 行。每轮被测核的全部 >100 ms 样本，保留没有前后完整边界的样本。`lateness_ms` 为超过期限的时间，`wall_ms` 包括预定睡眠，`thread_cpu_ms` 是该线程记账时间。`entry_skew_ms/exit_skew_ms` 为采样到的向量边界减程序边界，`last_vectors/next_vectors` 的加号表示同一采样格 |
| [已对时的调度和硬时钟窗口](context-windows.csv) | 147 行。每行一个完整 `single` 或 `dual_intersection` 事件。`witness_gap_ms` 是程序停顿，`interior_ms` 和 `own_current_hot_ms` 是裁剪后的中段及其中实验任务作为 current 的墙钟时长。`arch_timer_irq/sched_slices` 是该核中段内实测的 IRQ 和时间片增量。以 `normal_` 开头的是提前一秒的同核对照，缺少时留空。外围计时单独列在 `outer_*_ms` |
| [事前规则](holdout-plan.md) · [独立验证的相邻间隔](timing-holdout.csv) | 72 行，训练所用的两个早期阶段不在此表。两个预测时刻与实测下一起点都相对前一次起点，以 ms 计，`nearest_error_ms` 是二者中较小的绝对误差。`within_5ms` 为命中标志 |
| [首轮 26 条慢请求](original-slow-requests.csv) · [独立复测的 2,000 条请求](fresh-requests.csv.gz?raw=true) | 首轮表保留请求内主机计数不完整的 3 条。复测表每次重启 100 条，`ttft_ms` 为客户端首 token 延迟。各角色列是该请求内 CPU18 上就绪、少获执行与 TIMER 长窗的重合毫秒数，可能来自同一类内多条记录。不同角色可同时重合，请勿将各列相加当作端到端损失 |
| [关键线程实际调度](paired-placement.csv) · [运行时后台线程](runtime-placement.csv) | 关键线程是调度事件完整的 8 对重启，每版 800 条。每个 `early_hit_pct` 对应一个进程 100 条请求的前 400 ms。运行时角色细分是另选的 5 对，每版 500 条。两套表的分母各自独立 |
| [分项统计与覆盖](measurements.json) | 150 秒初筛、75 秒安静对照、100 秒向量实验、调度样本纳入/排除及丢失、独立预测及打乱对照。调度与预测摘要的四个分位数依次为最小值、中位数、p90、最大值。包含 HTTP 请求覆盖与按整对重启重抽得到的关键线程落点区间 |
| [双机热点图](independent-hosts.png) · [A 机中断时间线](irq-timeline.png) | 前者给出两机时间线与同核号比较，后者来自 A 上更早一轮独立定位。每张图使用所在实验自己的时间轴 |

## 复算

在保存了本目录的机器上运行。复算只用 Python 3 标准库，直接读取压缩 CSV，打印逐核长窗数及最大值、单/双线程窗口数、时间预测命中数、两版慢请求数和关键线程落点，并与分项统计核对。

```bash
sha256sum -c SHA256SUMS
python3 recompute.py
```

输出应包括 A·CPU18 29 次，B·CPU44 28 次，调度窗口 A 37+37、B 36+37，独立预测 A 36/37、B 35/35，复测 TTFT 超过 1 秒旧/新 9/12 条，关键线程早期切入旧/新 36/28 条。完整两台实机的 timer 回调采集仍待有宿主权限的一端执行。[复算程序](recompute.py)
