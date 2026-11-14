# write_report.py

原始验证文件快照；源码 SHA256：`e67ba66704fb5dcae34b2a3a2f161dfb678a1781e5665ba1d0a2ecd6aa6836f9`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
"""Generate the final follow-up report from closed runs and comparisons."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((ROOT/rel).read_text())
initial={m:read(f'results/compare-{m}-external-pr5030.json') for m in ['agents','sage']}
recheck=read('results/compare-agents-external-pr5030-recheck.json')
controls=read('results/local-buffer-control-summary.json')
lines=[
'Mooncake PR #5030 A3 复核报告，2026-11-13（Asia/Shanghai）',
'工作区：/root/workspace/a3-mooncake-pr5030-20261113',
'',
'本轮按用户提供的 PR #5030 与两份 A3 脚本补全了接入，指定 mooncake-transfer-engine-npu==0.3.13.post1 已完成实际 fabric 写读。Sage-A1.5（MTP3）和 Agents-A1（MTP关闭）均在迁移前后环境验证了跨实例外部 KV 命中、GDN 状态恢复及对应输出精度。此结论只覆盖下列实测范围。',
'',
'对上一轮结论的修正：此前只允许 protocol=ascend，仍缺少 PR 中的设备适配；SDK 还用了未对齐的64MiB segment，模型与SDK传入非零的小 local buffer。新配置在相同驱动25.5.6/CANN9.1.0上成功分配fabric内存并完成传输。因此，先前“需要先解决宿主fabric/驱动阻塞才能验证”的判断被本轮证据取代。厂商文档的HDK版本建议不能作为本机该路径不可用的实测结论。旧日志与旧证据包保持原样，作为失败配置的历史记录。',
'',
'参考与固定源码：',
'- https://github.com/InternLM/lmdeploy/pull/5030；读取时为open draft，head=1a09270b0835bb871be4eb4df70f9d04c8ccaec2，base=3f99ed1502ce3f7cf9747afc2138ff624e5d5457。元数据、diff及三个文件的固定版本已保存。',
'- /afs-weights/wangqing/kv_offloading_dev/dense_impl/serve_glm52_tpep_1015.sh',
'- /afs-weights/wangqing/kv_offloading_dev/dense_impl/launch_mooncake.sh',
'- 两者引用的mooncake_ascend.json及vllm_ascend_rc30_env.sh均已复制并记录SHA256。',
'- baseline：b334d7aaf8afd8fc5f77fcf34e3c5f028a2b7c9d，基于fb5de87a；配套dlinfer仍为3b9096341a41e9d067104a39d914e47ca2d7ec50。',
'- native：4930080caac39c73427fcfd083d082fdebf7ec09，基于已修复M1的6421cefe；不安装或导入dlinfer。',
'',
'两边应用完全相同的PR适配。唯一合并冲突是当前主线已把state_slot更新为state_destination；保留当前字段和compact-state逻辑，使用PR的设备事件。三个适配文件的哈希完全一致。独立本地分支、format-patch及固定版本见inventory/comparison-revisions.json和artifacts/pr5030-*.patch。没有推送远端。',
'',
'运行方式与验收边界：',
'新建两个独立venv，复用前轮冻结依赖，项目命名空间独立。Mooncake NPU wheel仍为0.3.13.post1，native主进程及worker有dlinfer导入阻断和审计。当前镜像仍存在原有pip check/依赖版本约束失败；这不是受支持版本组合的干净wheel安装验收。',
'',
'依照脚本设置ASCEND_ENABLE_USE_FABRIC_MEM=1、ACL_OP_INIT_MODE=1和CANN/ATB路径。connector extra_config显式device_backend=ascend。fabric使用1GiB整数倍的global segment，模型使用脚本中的8GiB；JSON local_buffer_size=4GiB在PR fabric分支被有效覆盖为0。不注入外部TransferEngine、不做客户端HBM注册，保存/加载的同步事件使用torch.npu.Event。',
'',
'master由本轮管理，127.0.0.1:55061，metrics55062，eviction高水位0.9/比例0.1、lease30000、client_ttl120、enable_offload=false。使用该wheel支持的--rpc_port替代已弃用别名--port。没有连接同事的master。',
'',
'模型仍用前轮相同权重与TP2/BF16/block128/prefill8192/session262144/cache0.95/prefix state budget16/alignment128/last-only；Sage MTP3、Agents MTP关闭。文本推理、thinking关闭、greedy、seed42、最多32输出token并遵守EOS。没有把GLM脚本的TP16/EP16/W8A8移植成两模型的测试配置。',
'',
'独立SDK实测：',
'| 场景 | 结果 |',
'| --- | --- |']
for label in ['baseline-pr5030-fabric-sdk-batch','native-pr5030-fabric-sdk-batch','native-pr5030-fabric-sdk-8g']:
    data=read(f'results/{label}.json');events=data['events']
    writes=sum(e['event']=='put' for e in events);reads=sum(e['event']=='get' for e in events)
    lines.append(f"| {label} | complete={data['complete']}，{writes}次写/{reads}次读，子进程退出码{data['child_exitcodes']} |")
lines += ['',
'每组两个真实NPU进程，双向128KiB/2MiB/8MiB，单对象拆成三个NPU地址片段，通过connector使用的batch_put_from_multi_buffers/batch_get_into_multi_buffers传输。每次返回字节数正确，目标缓冲区的逐字节比较和SHA256一致；包含独立进程关闭。SDK探针不是生产带宽基准。',
'',
'保留的失败控制：最初标量put_from成功，但get_into在local_buffer_size=0时返回-200，因为该API需要临时client buffer。它与connector实际使用的批量散聚接口不同，不能将此失败视为批量fabric路径失败；也没有删除这次失败结果。',
'',
'同运行时、同1GiB global segment的local-buffer单变量控制：']
for item in controls:
    lines.append(f"- effective local={item['local_mib']}MiB：setup_success={item['setup_success']}，退出码{item['returncode']}，aclrtMapMem失败={item['map_mem_failure']}。")
lines += ['',
'模型外部缓存证据：producer在逻辑0,1上完成四条不同长请求并保持存活。consumer在2,3上重新创建引擎，先做独立冷预热，再读取相同四条请求。consumer没有这些请求的本地prefix状态。各共享请求实际7498输入token、8输出token，取回7424token的KV及其状态，每个rank加载59个Store对象。Agents每rank读取108462080字节，其中GDN状态32440320字节；Sage每rank读取211173376字节，其中状态127549440字节，包含其MTP配置下的状态布局。日志核对了两个rank，保存完整Ray worker日志以避免日志去重掩盖次数。',
'',
'| 模型/环境 | 外部缓存命中 | 冷→外部精度 | 成功加载批次数 | 生命周期 |',
'| --- | --- | --- | --- | --- |']
for model,comp in initial.items():
    for side in ['baseline','native']:
        x=comp[side];pairs=list(x['cold_to_external'].values())
        assert x['complete']
        lines.append(f"| {model}/{side} | {sum(v['external_cached']==7424 for v in pairs)}/{len(pairs)}请求 | token精确={all(v['exact_tokens'] for v in pairs)}；{sum(v['score_count'] for v in pairs)}个分数，max差{max(v['max_abs'] for v in pairs)} | {x['logs']['consumer']['successful_load_batches']} | 子进程{x['child_exitcodes']} |")
for model,comp in initial.items():
    rows=[v for r in comp['migration_comparison'].values() for v in r.values()]
    lines.append(f"{model}迁移前后：{sum(v['exact_tokens'] for v in rows)}/{len(rows)}响应token一致，{sum(v['score_count'] for v in rows)}个共同历史分数max差{max(v['max_abs'] for v in rows)}，严格精度判定={comp['migration_precision_pass']}。这里包括producer四条和consumer五条（含独立预热）；四条外部命中才计入上表。")
lines += ['',
'C1冷/外部缓存延迟（不含模型加载；每组先预热，首组3个正式请求）：',
'| 场景 | 冷TTFT中位数 B→N(ms) | 外部TTFT中位数 B→N(ms) | 外部TTFT变化 | 外部请求总时长变化 |',
'| --- | --- | --- | --- | --- |']
for name,comp in [('Agents首组3次',initial['agents']),('Sage首组3次',initial['sage']),('Agents反向复测15次',recheck)]:
    b=comp['baseline']['timing'];n=comp['native']['timing'];delta=comp['timing_change_percent']
    lines.append(f"| {name} | {b['cold_ttft']*1000:.1f}→{n['cold_ttft']*1000:.1f} | {b['external_ttft']*1000:.1f}→{n['external_ttft']*1000:.1f} | {delta['external_ttft']:+.2f}% | {delta['external_elapsed']:+.2f}% |")
first=initial['agents']['timing_change_percent']['external_ttft'];second=recheck['timing_change_percent']['external_ttft']
lines += ['',f'Agents首组外部TTFT上升{first:.2f}%触发预设5%阈值，随后以native→baseline反向顺序、15个不同prefix正式请求复核，变化为{second:+.2f}%。'+
    ('两轮都超过5%，保留为重复性能退化信号，尚未定位具体代码原因。' if second>5 else '该项超过5%的退化没有在扩大样本复测中重复，不将首组三次的小差异定为稳定回归。'),
    f"扩大样本中，baseline/native外部功能分别为{recheck['baseline']['external_function_pass']}/{recheck['native']['external_function_pass']}，迁移严格精度判定={recheck['migration_precision_pass']}。原始逐请求时间、token及分数全部保留。",
'',
'这里测的是约7.5k输入、8token检索输出、C1场景。冷与外部请求使用不同NPU对，迁移前后严格保持角色与设备一致；因此同分支冷/热比值包含设备对差异，不能当成隔离设备因素的纯传输加速比。没有据此宣称长生成吞吐、C4/C16、DP/EP、多机或生产长期稳定性通过。上一轮HTTP C4退化信号及Sage MTP长生成等价性失败仍保留，未被本轮短检索通过覆盖。',
'',
'CPU回归与未覆盖项目：']
lines += [
'按用户后续要求，两套隔离环境均补齐tilelang==0.1.15及配套依赖：apache-tvm-ffi==0.1.12、torch-c-dlpack-ext==0.1.5、ml-dtypes==0.6.0、z3-solver==4.15.4.0。使用aarch64预编译wheel，SHA256逐一核验后离线安装。TVM FFI由这两套环境原来链接的0.1.14.post1切换至TileLang要求范围内的0.1.12；此前冻结依赖和原镜像安装保持原样，未修改项目源码。',
'两边TileLang导入、XGrammar实际语法解析及新增依赖约束检查均通过，pip check未新增错误；镜像已有的依赖约束问题仍保留。native再次确认无dlinfer发行包、无已导入模块及无导入阻断事件。详见inventory/tilelang-followup中的版本、freeze、链接变更和审计记录。']
for side in ['baseline','native']:
    cases=list(ET.parse(ROOT/'results'/f'{side}-pr5030-worker-unit-tilelang.xml').iter('testcase'))
    failed=[c for c in cases if c.find('failure') is not None or c.find('error') is not None]
    skipped=[c for c in cases if c.find('skipped') is not None]
    lines.append(f'- {side} worker suite：{len(cases)-len(failed)-len(skipped)}通过、{len(failed)}失败、{len(skipped)}跳过；结果见results/{side}-pr5030-worker-unit-tilelang.xml。最初缺少TileLang时的100通过/1失败XML与日志仍保留。')
lines += [
'- 原先失败的test_glm_mtp_store_roundtrip_restores_joint_cache_without_raw_tails已在两套完整套件中通过，native还单独复测通过。该项验证CPU BytesStore中的GLM/MTP联合缓存往返与布局恢复，不代表CUDA内核执行或GLM整模型推理通过。本次依赖补齐后未重新运行前述NPU模型测试，前述模型数据仍对应补齐前记录的环境。',
'- 未测GLM5.2量化模型、GLM MLA/DSA整模型、TP16/EP16、跨机fabric、其他Mooncake版本、非fabric Ascend协议、SSD offload、生产压力/故障恢复及16GiB segment关闭行为。',
'- 同事checkout中的未提交store-retirement处理已记录；本轮8GiB数据传输后均实测正常close，使用的是固定PR提交的生命周期代码。',
'- 安装、数值、功能、性能、生命周期分别判定；跳过、失败和未覆盖没有计入通过。',
'',
'复现与交付：REPRODUCE.txt提供复用环境、SDK、模型双实例及TileLang补齐与测试命令。logs/*.command.json记录命令、环境、源码、harness快照与退出码；artifacts/ray-logs保留完整rank日志；results/compare-*.json为数值/时间汇总，configs为真实Store配置。inventory/preservation-final.json和cleanup-final.json记录最终保护与资源清理审计。更新后的evidence.tar.gz包含五个依赖wheel及逐文件SHA256；补齐依赖前的证据包另存为evidence-before-tilelang.tar.gz，原始SHA256保持不变。']
(ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print('Saved REPORT.txt')

```
