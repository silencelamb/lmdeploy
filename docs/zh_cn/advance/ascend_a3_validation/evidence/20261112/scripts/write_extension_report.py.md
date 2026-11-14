# write_extension_report.py

原始验证文件快照；源码 SHA256：`54c99490c30cbbac734e26b0b12cfc1d1dc72b17a4d6208f2b16a5ac72b462a9`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lines=['用户追加的 Agents-A1 与 Mooncake 0.3.13.post1 验证',
    '',
    '扩展判据在执行前写入 ACCEPTANCE-EXTENSION.txt；两份追加的部署文档已复制到 inventory。Agents-A1 使用 /afs-weights/Agents-A1 的同一份 BF16 权重、TP2、session262144、block128、batch16、prefix budget16/alignment128/last-only。按其 A3 文档关闭 MTP；Sage 部署对比保持 MTP3。Agents 的完整模型 SHA256/配置清单已保存，结束时文件大小和 mtime 不变。',
    '',
    '| Agents-A1 场景 | 实际结果 | 判定 / results/证据 |',
    '| --- | --- | --- |',
    '| eager 长短请求与重复请求 | 21/21 序列、1344 token、6720 分数完全一致 | 通过；compare-agents-eager.json |',
    '| graph+prefix 质量与固定历史评分 | 16 个生成响应/56 token/280 分数一致，12/12 确定答案；4 个评分请求123行/635分数差0，top1无差异 | 通过；compare-agents-graph-prefix-extra.json |',
    '| 真实 prefix 复用与干扰请求 | 7490输入token，重复命中7424token，插入两个其他请求后检索和输出仍一致 | 通过；*-agents-graph-prefix-extra.json |',
    '| 近上下文上限检索 | 261845输入token、8输出token、40分数一致，均得到ORCHID-8246；单次36.585s/38.201s | 此单条场景通过；compare-agents-context-boundary.json |',
    '',
    '以上覆盖实际数值、缓存状态和边界长度，不代表完成 GPQA 82.3 指标、全部 agent/tool-use 任务或多模态精度。短题检查和单条长上下文不能外推到完整模型评测。',
    '',
    'Agents HTTP 首组按 baseline→native，每档预热后5轮，C1/C4/C16；C1延迟与C4吞吐触发>5%复核，因此追加 native→baseline、C1/C4各7轮。固定128输出token及相同请求，指标定义与Sage段相同。',
    '',
    '| 组别/并发 | 总吞吐 B→N | 变化 | TTFT p50 B→N(s) | TTFT p95变化 | 输出速度p50变化 | SLO请求 B / N |',
    '| --- | --- | --- | --- | --- | --- | --- |']
all_data={}
for name,label in [('compare-http-agents','首组5轮'),('compare-http-agents-recheck','反向7轮')]:
    d=json.loads((ROOT/'results'/f'{name}.json').read_text());all_data[name]=d
    for c,r in d['scenarios'].items():
        b=r['baseline'];n=r['native'];delta=r['change_percent']
        lines.append(f"| {label}/C{c} | {b['median_throughput']:.2f}→{n['median_throughput']:.2f} | {delta['median_throughput']:+.2f}% | {b['ttft_p50']:.3f}→{n['ttft_p50']:.3f} | {delta['ttft_p95']:+.2f}% | {delta['output_tps_p50']:+.2f}% | {b['slo_pass_requests']}/{b['requests']} / {n['slo_pass_requests']}/{n['requests']} |")
lines.extend(['','Agents 的 C4 总吞吐首组下降13.68%，反向复测下降2.67%，没有重复确认超过5%的吞吐回归；但逐请求流式输出速度p50两组下降7.13%和8.99%，保留为可重复性能退化信号。C1的TTFT p95两组分别上升5.96%和5.07%（约39ms和35ms），属于刚超过阈值的尾部延迟退化信号，小样本分位数的局限保留。C1吞吐两组都在5%内；C16本次5轮没有测到性能下降，但新旧SLO均为0/80。C4 SLO首组17/20→11/20，复测22/28→21/28。性能不能整体判通过，具体迁移代码归因仍未完成。针对API调度影响，另做同prompt、同输入/输出长度、固定C4批次的pipeline对照，其结果单列，不覆盖HTTP证据。','',
    'Mooncake 使用指定的 mooncake-transfer-engine-npu==0.3.13.post1，已安装到新增 baseline-mooncake/native-mooncake 两个环境。Python3.12/aarch64 wheel 为15706817字节，SHA256 ba134f2cc99784aa32404c3a1406e3c85400792fb2c62bb458cd4757a07cc4bb。原三个环境仍保留镜像的0.3.11.post1，未被升级。新增 native 环境也没有安装或导入 dlinfer，审计见 inventory/*-mooncake-environment.json。',
    '',
    '按 GLM/H200 文档采用 embedded MooncakeStoreConnector、kv_role=kv_both、P2PHANDSHAKE 和单独 master。A3 传输选 protocol=ascend，开启 ASCEND_ENABLE_USE_FABRIC_MEM=1；使用本次管理的127.0.0.1:55051/55052服务，没有连接 /afs-weights/mooncake.json 中的外部 master。A3 验证配置将每worker全局segment设为2GB、local buffer设为256MB，其余模型配置与各自部署对比一致。该容量用于代表性功能验证，不能等同H200文档160GB/4GB部署的容量或性能。',
    '',
    '发现两边原代码都拒绝 protocol=ascend，仅允许rdma/tcp。这是共有平台接入缺口，已在额外分支允许ascend并验证配置原样传给SDK。两个环境配置/初始化测试各26 passed、77 deselected，属于CPU/stub测试。baseline额外提交e285fa49d1759d709660588764ce4091566240af，native额外提交9f76d08a26b484b5bd23391ffb829a9f50c4f635；原main/native比较工作区不变。补丁只改协议校验及参数传递测试，没有更改kernel或传输算法；不能用这项通过替代真实数据传输。',
    '',
    '| Mooncake 实测 | 结果 | 归类 |',
    '| --- | --- | --- |',
    '| NPU wheel安装与API导入 | 新旧扩展环境均为0.3.13.post1 | 通过，安装事实 |',
    '| 独立SDK、两个NPU进程、fabric开启 | 两个环境均在Store.setup失败；64MiB segment/16MiB local尚未加载模型就出现aclrtMapMem错误及std::bad_alloc | 共有运行环境阻塞；零次有效数据写读 |',
    '| 设置fabric变量为0的控制 | 仍显示use_fabric_mem=true，同样失败 | 该版本只判断变量存在，0不等于关闭 |',
    '| 真正unset fabric变量的SDK控制 | ADXL初始化失败503900，无法从hccn.conf/hccn_tool读取NPU IP | 另一项环境阻塞；不计非fabric传输通过 |',
    '| Sage MTP3+Store，baseline/native | 两边实际加载模型并进入缓存引擎构建，Store.setup的fabric映射失败，rc1 | 2个启动场景失败，精度/命中/性能未验证 |',
    '| Agents MTP关闭+Store，baseline/native | 同一Store.setup的fabric映射失败，rc1 | 2个启动场景失败，精度/命中/性能未验证 |',
    '',
    '四个模型运行均保存 complete=false 的结果和完整栈，见 results/mooncake-model-progress.json、*-fabric-model.json 和 logs/*-fabric-model.log。计划中的跨引擎外部cache hit、NPU缓冲区写读一致性、GDN状态恢复、缓存污染隔离与Store冷/热性能全部未完成，不作通过判断。错误在小内存、无模型的SDK探针中也存在，不能解释为迁移新增的模型OOM。',
    '',
    '该版本的官方Ascend Direct文档要求fabric memory模式使用CANN9.0+、HDK26.0+；当前CANN9.1.0，而npu-smi board实际报告Software Version25.5.6、Firmware7.8.0.10.220。实际驱动映射失败与这项版本缺口同时存在；不能仅凭此错误断言升级驱动是唯一所需修复。容器内也没有找到/etc/hccn.conf或hccn_tool，npu-smi的IP查询明确不支持；未猜测或修改NPU网卡地址，已向用户询问是否有其他挂载路径。当前证据不足以继续验证Store数据传输。',
    '',
    '本次NPU wheel实际走Ascend Direct/HIXL，日志明确显示fabric选项。镜像已有memfabric_hybrid1.2.0，但没有把“该包已安装”当作fabric传输已通过。单独的mooncake-transfer-engine-memfabric在查询时只有0.3.14，没有擅自替换用户指定的0.3.13.post1。源代码还确认环境变量只要存在就打开fabric，关闭必须unset。官方依据及固定commit链接在 inventory/mooncake-reference-sources.txt，wheel hash与源码快照已保存。',
    '',
    '继续完成Mooncake验证需要宿主提供可用的fabric内存映射能力和匹配的驱动/运行时，或先补齐真实NPU网卡配置以验证非fabric路径。补齐后应从独立SDK的实际NPU写读校验开始，再验证两个模型的跨引擎Store命中及同配置精度/性能；本次不将这些尚未执行的后续步骤描述为成功。'])
text='\n'.join(lines)+'\n'
control_path=ROOT/'results/compare-agents-c4-pipeline-control.json'
if control_path.exists():
    control=json.loads(control_path.read_text())['bench']['4']
    b=control['baseline'];n=control['native'];delta=control['change_percent']
    paragraph=(f"追加的固定C4批次pipeline对照共7轮/28请求，使用与HTTP复测完全相同的prompt、171输入token、128输出token、MTP关闭、thinking关闭。新旧28/28 token序列一致，两种入口内及跨入口的28/28最终文本也全部一致。总吞吐{b['median_throughput']:.2f}→{n['median_throughput']:.2f}token/s（{delta['median_throughput']:+.2f}%），输出速度p50 {b['output_tps_p50']:.2f}→{n['output_tps_p50']:.2f}（{delta['output_tps_p50']:+.2f}%），TTFT p50 {b['ttft_p50']:.3f}→{n['ttft_p50']:.3f}s。本组未复现下降，说明不能把HTTP观测直接外推为固定批次引擎计算的回归；HTTP调度/流式路径中的重复退化信号仍保留，具体代码归因未完成。证据compare-agents-c4-pipeline-control.json、agents-c4-control-output-audit.json。\n\n")
    text=text.replace('Mooncake 使用指定的',paragraph+'Mooncake 使用指定的')
(ROOT/'results/extension-report.txt').write_text(text)
path=ROOT/'REPORT.txt';report=path.read_text()
marker='\n\n用户追加的 Agents-A1 与 Mooncake 0.3.13.post1 验证\n'
if marker in report:report=report.split(marker)[0]
path.write_text(report+'\n\n'+text)
print('Saved extension section with measured performance and explicit Mooncake failures.')

```
