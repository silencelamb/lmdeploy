# Ray与主机尾延迟独立调查

更新日期：2026-11-17，Asia/Shanghai。本文将Ray/主机停顿从GDN正确性修复中单独整理；依据既有A3/H200证据，没有在本轮新增运行、改动Ray或部署规避。

## 当前结论

**问题仍开放。** A3约半秒停顿可由无Torch/NPU的纯Ray和无Python/Ray的C++程序复现。已记录的关键事件对应CPU18上可运行线程长时间等待，同时存在长softirq/TASKLET活动。具体callback、来源IRQ/模块及约2.56秒周期的触发源还没有完成宿主机取证，不能把所有尾延迟统称为同一个根因，也不能归因于ARM指令集。

H200/x86两个6000次CPU-only对照没有观测到>50ms join，是有限阴性对照，不是永久修复或架构根因证明。本调查不混入GPU模型性能计时，不改变已独立验证的GDN/MTP编译器修复结论。

## 从服务停顿到主机调度证据

服务和worker跟踪记录了约484ms的HTTP流式间隔；norm重构性能复查中245–551ms间隔主要伴随222–533ms等待前次Ray worker提交RPC完成。纯Ray探针保持submit→sleep12ms→join→free，并记录投递、对象就绪、方法入口/出口、结果回调和event-loop。

原ray.init可能加载ACL资源探测，后续显式resources={'NPU':0}并核对进程maps，driver/worker无Torch、torch_npu或ACL时仍能复现。用户进程不加载ACL不能排除宿主机驱动。result_transport包含固定sleep期间，单rank的future_to_join也可能等待其他rank，不能直接当作纯网络延迟。

| 关键样本 | 受影响线程 | join等待 | runqueue增量 |
| --- | --- | --- | --- |
| 479 | driver worker.io | 542.033ms | 543.177ms |
| 1322 | driver gRPC nexting_thread | 522.367ms | 532.658ms |
| 1826 | actor执行线程 | 512.737ms | 521.472ms |
| 1984 | actor RPC线程 | 341.133ms | 351.582ms |
| 2330 | actor Python event-loop | 534.651ms | 551.290ms |

这些样本的慢区间采样到线程处于R状态、CPU18。时间窗口可能包含相邻事件，不将所有计数增量机械分摊给一个RPC。[关键样本](evidence/mtp-ray-deep-20261116/results/ray-cpu18-critical-samples-r1.json)、[调度窗口](evidence/mtp-ray-deep-20261116/results/ray-sched-all-r0-sched-summary.json)。

## 无Ray的C++控制

复现缩成两个pipe子进程、8KiB输入、12ms nanosleep、读取确认，没有Python、Ray、模型或设备调用。

| 配置 | 迭代 | 实际sleep>50ms | 最大sleep | 停顿起点间隔中位数 |
| --- | --- | --- | --- | --- |
| CPU18，r0 | 2500 | 15 | 565.844ms | 2.5614s |
| CPU18，r1 | 2500 | 14 | 573.717ms | 2.5605s |
| CPU18，r2 | 1250 | 8 | 571.409ms | 2.5579s |
| CPU42，相同程序 | 2500 | 0 | 21.413ms | — |
| 不绑定、子进程nice15 | 6000 | 0 | 14.461ms | — |

独立CPU42监控器观测到：559.312ms唤醒延迟窗口内，CPU18增加55个softirq tick（CLK_TCK100，约550ms），TASKLET增加1；其他长窗口约52–57个softirq tick。[对应数据](evidence/mtp-ray-deep-20261116/results/cpu18-stall-correlation-r1.json)。这支持CPU18的宿主机执行干扰是所捕获停顿的直接机制，尚不足以确定具体驱动回调。

CPU18的IRQ清单涉及NVMe、SAS、消息通知等，名字和计数不足以指认责任模块。[IRQ清单](evidence/mtp-ray-deep-20261116/inventory/irq-cpu18-r0.json)。另有107/153ms残余等待与诊断探针自身generation-2 GC重合；只解释对应事件，不能把全部尾延迟归为GC或据此关闭生产GC。

## A3进程绑核：已测设置与复现命令

绑核只作用于启动的命令及其新建子进程，不修改全局IRQ亲和性，也不改变Ray的await、同步顺序或计算算法。实测使用两种设置：限制到CPU40–79；或者从进程当前允许的CPU集合中仅移除CPU18。后者保留了原来就不可用的CPU，例如HTTP记录中CPU172原本也不在允许集合内，因此不能简单把0–319全部重新放开。

| CPU-only控制 | join数 | >50ms | >300ms | 最大join |
| --- | --- | --- | --- | --- |
| 默认配置 | 6000 | 16 | 14 | 574.971ms |
| CPU40–79 | 6000 | 0 | 0 | 46.783ms |
| 仅从允许集合排除18，r0 | 6000 | 2 | 0 | 138.592ms |
| 仅从允许集合排除18，r1 | 6000 | 3 | 0 | 120.755ms |
| 排除18并记录GC | 6000 | 2 | 0 | 152.603ms |

原生Agents-A1、非MTP、TP2、真实HTTP C1/C4各6轮加预热的排除18控制完成30条测量请求和5条预热；3618次RPC join无>50ms，稳定阶段无>100ms SSE间隔。剩余CPU探针107/153ms等待与探针自身GC有关；不以关闭生产GC作为修复。[HTTP缓解证据](evidence/mtp-ray-deep-20261116/results/http-exclude18-r0-summary.json)、[原始CPU控制与边界](mtp_ray_investigation_20261116.md#本机缓解验证与剩余小停顿)。

已交付的[launch_affinity.py源码](evidence/mtp-ray-deep-20261116/scripts/launch_affinity.py.md)先读取`os.sched_getaffinity(0)`，减去指定CPU后调用`os.sched_setaffinity`，再exec原命令。原A3工作目录上的用法如下（其余原启动参数照常传入）：

```bash
python /root/workspace/a3-mtp-ray-investigation-20261116/scripts/launch_affinity.py \
  --exclude 18 -- \
  lmdeploy serve api_server "$MODEL_PATH" --backend pytorch --device ascend --tp 2
```

不依赖该脚本时，等价地先计算允许集合，再用taskset启动。以下仍是针对已确认CPU18干扰的A3主机示例：

```bash
A3_RAY_CPU_LIST=$(python -c 'import os; cpus=sorted(os.sched_getaffinity(0)-{18}); assert cpus; print(",".join(map(str,cpus)))')
taskset -c "$A3_RAY_CPU_LIST" \
  lmdeploy serve api_server "$MODEL_PATH" --backend pytorch --device ascend --tp 2
```

如果选用CPU40–79，也应先与当前允许集合求交。驱动进程必须在初始化本任务Ray runtime/worker之前设置亲和性；连接已存在的其他Ray集群不会自动改变那些worker的设置。用`taskset -pc PID`及各线程的`Cpus_allowed_list`核对实际driver/raylet/worker，不能仅根据父进程命令推断。只管理本任务实例，不使用全局ray stop去终止其他用户任务。

这些是本机有限窗口的临时缓解/诊断设置，未作为迁移PR默认策略交付，也没有复制到H200模型验证。具体callback和永久修复仍待取证。

## H200/x86独立CPU对照

无Torch/NPU环境：Python3.12.14、Ray2.48.0、NumPy2.2.6；复现脚本SHA256为870843b45f906a203ef8639c15d041bed878ea3489e1895889de737b8c8583b5，与参考逐字节相同。保留原提交/等待/确认顺序及默认亲和性，不照搬CPU18避让。

| 全新进程 | 迭代 | 最大put join | 最大direct join | >50ms join |
| --- | --- | --- | --- | --- |
| 1 | 6000，put/direct各3000 | 45.123ms | 1.444ms | 未观测到 |
| 2 | 6000，put/direct各3000 | 40.834ms | 2.078ms | 未观测到 |

A3参考Python为3.12.13，故不能称软硬件完全相同。此对照只表示本轮窗口未复现，不证明H200永久免疫、ARM指令集根因或Ray修复。原始投递/确认/event-loop、进程maps、schedstat与softirq信息保存在[H200 Ray证据](evidence/cuda-h200-20261116/ray-summary-audit.json)及[GPU报告对应章节](cuda_regression_20261116.md#ray-独立-cpu-对照)。插桩GPU模型退出挂起另列，未与此CPU问题合并。

## 剩余取证与完成判据

容器缺少tracefs/内核地址与栈等权限，已有[权限记录](evidence/mtp-ray-ir-host-20261116/inventory/host2-permission-limits-r0.json)。下一步应在有权限的宿主机使用已交付的[tasklet采集工具](https://github.com/silencelamb/lmdeploy/tree/f9ab50d877c384bfb04ae43afa6af0c388fc451b/tools/ascend_a3/diagnostics)，而不是反复运行信息增量很小的相同容器测试。

1. 通过BTF和tasklet调度发现callback，再关联callback入口/退出、softirq、IRQ及可选perf栈；记录符号/模块、耗时、对应停顿、丢事件和probe miss。
2. 找到周期触发源后做单变量可恢复控制，证明长回调、runqueue等待和请求停顿因果关联，避免按IRQ名字猜根因。
3. 对齐脚本、Python补丁版、Ray/NumPy、输入、观测窗口及探针开销，补跨主机逐事件比较。
4. 在责任层独立分支修复，用相同默认亲和性复测C++、纯Ray和HTTP C1/C4，至少三轮全新进程；剩余host调度、GC、对象就绪和内部队列事件分别记录。

完成标准为“具体回调/触发源→可重复因果控制→责任层修复→同配置回归”。避开CPU18、关闭GC、删除await、改同步顺序或单次没有长尾均不算完成；这些规避没有作为迁移PR修复交付。

更完整的历史命令、失败迭代、工具检查与采集限制保留在[原调查记录](mtp_ray_investigation_20261116.md#ray-后续调查计划)及[norm性能定位](gated_norm_performance_20261116.md)。
