# write_sage_performance.py

原始验证文件快照；源码 SHA256：`7097a0dfa2842e303815b72f80fbf33c8de2e81b515a7c7a78678d40547f5195`。

此文件用于复现和审计，原样保留当时的路径、版本和已失败运行。

```python
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
lines=['Sage-A1.5 HTTP 性能：TP2/MTP3/prefix，与前述部署参数相同。首组 baseline→native 每档预热后各 5 轮；复测 native→baseline 每档各 7 轮。每轮并发 1/4/16，实际输入 171–173 token、固定输出 128 token。新旧在各组使用完全相同的 server 参数、prompt 和实际输入/输出 token 数，比较脚本有断言。测量期间没有并行模型测试。',
    '', '| 组别/并发 | 中位总吞吐 B→N (token/s) | 变化 | TTFT p50 B→N (s) | 流式输出速度 p50 B→N | SLO请求 B / N |',
    '| --- | --- | --- | --- | --- | --- |']
for name,label in [('compare-http-fresh','首组5轮'),('compare-http-recheck','反向7轮')]:
    data=json.loads((ROOT/'results'/f'{name}.json').read_text())
    for c,r in data['scenarios'].items():
        b=r['baseline'];n=r['native']
        lines.append(f"| {label}/C{c} | {b['median_throughput']:.2f}→{n['median_throughput']:.2f} | {r['change_percent']['median_throughput']:+.2f}% | {b['ttft_p50']:.3f}→{n['ttft_p50']:.3f} | {b['output_tps_p50']:.2f}→{n['output_tps_p50']:.2f} | {b['slo_pass_requests']}/{b['requests']} / {n['slo_pass_requests']}/{n['requests']} |")
lines.extend(['',
    '吞吐的首组 C1/C4 下降在反向复测中没有重复出现，不能把 -15.57%/-6.39% 当作稳定吞吐回归，也不能只取第二组较好结果宣称全通过。C4 的逐请求流式输出速度 p50 在两组分别下降 12.18% 和 5.87%，按事先阈值保留为重复的性能退化信号。它与 MTP 非重复性、各请求接受长度和调度的关系尚未完成归因，不能据此断言具体迁移代码造成。C16 两组总吞吐下降 2.98% 和 0.47%，均在 5% 内；首组输出速度 p50 下降 5.37% 未在复测重复。整体性能结论不是全通过。',
    '',
    'SLO 只针对本组短提示：C1 两边全部达到；C4 有未达标请求；C16 新旧都没有请求同时达到 TTFT<=3s 与输出>=40token/s。这是当前单实例负载的实测，不代表四实例/router goodput，也不能把基线已不达标归为迁移新增。',
    '',
    '吞吐为总输出 token/批次完整 wall time，含 prefill 与 HTTP；TTFT 为首次非空 SSE。流式输出速度使用 (completion_tokens-1)/(最后非空SSE时间-首次非空SSE时间)，MTP 的多 token 突发使它成为流式近似指标，不能解释为逐 token kernel 延迟。原始 SSE 时间和文本全部保留。初始 pipeline 3 轮的总吞吐变化为 C1 -3.59%、C4 +6.02%、C16 -0.73%，只作补充，不替代两组 HTTP 结果。',
    '',
    '反向复测的 npu-smi 采样中，两个环境物理芯片 0/1 的最高总 HBM 用量均为 61746/61397 MB；这是约4秒间隔采样、含驱动驻留占用，不能当作精确 allocator 峰值。启动/加载/编译单独保留，不计稳态吞吐。证据：compare-http-fresh.json/.csv、compare-http-recheck.json/.csv、npu-memory-samples.json 和对应 *http*-server.log。'])
text='\n'.join(lines)
(ROOT/'results/sage-performance.txt').write_text(text+'\n')
path=ROOT/'REPORT.txt';report=path.read_text()
pending='HTTP 性能数据将在反向配对复测完成后写入本位置。\nPERFORMANCE_PENDING'
if pending in report:path.write_text(report.replace(pending,text))
print('Saved Sage HTTP performance section')

```
