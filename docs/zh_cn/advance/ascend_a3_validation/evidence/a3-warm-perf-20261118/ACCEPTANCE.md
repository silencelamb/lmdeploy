# A3充分预热性能复核：预先判据

本轮于2026-11-18（Asia/Shanghai）启动。pilot只验证工具、形状覆盖和观测开销，不计正式性能验收。
CI通过不替代性能验收；不得以“没有显著差异”代替“已经证明等效”。

## 比较边界

- 历史迁移信号：LMDeploy `fb5de87a45df9999a19a569ed9441df8edfa6975` + dlinfer
  `3b9096341a41e9d067104a39d914e47ca2d7ec50` 与 native `6421cefea94edd350e419f556158639556357a27`。
- 当前交付：上述旧LMDeploy+dlinfer与当前PR `2ccf38a43f0fdc7415ce4ce9f177852bdacc9966`。
- norm隔离：已原生迁移的 `342db85ecff3c0dc864a4bf8613c589c270f4141` 与
  `5f1b1b3c12b1c198813f65275f778c7ad292fb02`。不得拿它代替旧dlinfer→native HTTP对照。
- CANN保持9.1.0；原环境与旧证据不改。新独立Python命名空间只读复用固定依赖，不对共享依赖pip安装。
  原生各环境不得安装/导入dlinfer。原编译器3.2.2/NPUIR1.2和正确编译器3.2.2.post2/NPUIR2.0分开命名。
- Sage MTP旧编译器有已定位的状态错误，不能把不正确/不自重复的输出用作性能等效认证。
  当前交付比较优先使用两侧共同的已验证NPUIR2.0配置；这属于明确的共同平台修正，不能称“只改预热”。
  原始旧编译器历史数字保留，不用新条件改写其测量事实。

## 覆盖项和负载

- HTTP：Agents非MTP、Sage MTP3，各自C1/C4。覆盖总吞吐、流式输出速度p50、TTFT p50/p95、TPOT。
  使用同模型、TP2、BF16、graph、prefix ON、session262144、prefill8192、block128、batch16、cache0.95，
  固定128输出token、greedy、seed42、thinking OFF、相同逐对prompt和token长度。
  流式速度保持历史定义 `(output_tokens-1)/(last_nonempty_SSE-first_nonempty_SSE)`，不混用算子时间。
- norm：Agents C1/C4的256输出token（同时覆盖原C4 TTFT信号）、Sage C1/C4的64输出token。
  同原norm配置：TP2/BF16/graph、prefix OFF、session8192、prefill256、batch8、cache0.5；保留原thinking默认行为。
  这是完整pipeline对照，不能代替HTTP指标。
- 主结果使用相同物理NPU0/1、串行运行、默认CPU亲和性；不关闭GC、不删除await、不改变RPC同步顺序。
  所有慢样本保留。CPU18避让若用于诊断，单独标注，不能作为主验收的替代。

## 预热和正确性资格

- 每个全新服务/引擎先覆盖C1/2/3/4，至少5个完整sweep，最多10个。
  最后连续两个sweep须无新forward形状、无新graph capture、无Triton编译/缓存加载调用、无autotuner bench。
  观测范围和厂商内部不可见的缓存限制如实说明；仅健康检查通过或一次warmup不足以声明稳定。
- 观测仅访问CPU形状信息/已经传回CPU的输出，不能增加NPU同步或设备到CPU拷贝。
  每个请求保存输入/输出token IDs，MTP另记录输出槽数和target/draft forward统计；不能沿分歧历史比较分数。
- 固定资格prompt在每个新进程各运行一次（避免同进程跨prefix命中比较）；用于跨进程自重复及版本间一致性检查。
  测量prompt均在早期token位置不同，记录实际cache hit；不把预热prompt的cache-hit结果混入主cache-miss负载。
- 测量阶段若仍出现编译/捕获/新形状，该次保留为预热不完整的失败记录，修正覆盖后使用新label复测；
  不可无限重标慢样本为预热。输出不正确、工作负载不一致的配置不能被计为性能通过。
- 相同输出下的批处理或MTP接受效率变化属于需解释的真实执行差异，不能删除这些慢样本来制造持平。

## 正式统计

- 每个比较至少8个独立的进程配对block，AB/BA各4组，同一对两侧prompt完全一致。
  每侧每block至少25轮C1和8轮C4；C1每侧合计不少于200请求，C4请求按batch/进程簇处理。
  进程启动和全部预热不计入测量，但单列保留时长。
- 以进程配对block为独立单位，对每个block的同指标计算新/旧变化；报告逐对值、配对中位数及95%配对bootstrap区间。
  同时报告两侧汇总值和其比值，不能挑两种统计中较有利者。p95额外给出保持block配对的总体分位数bootstrap区间。
- **等效（持平）**：工作负载资格通过，预先指定关键指标的95%区间均落入±5%。
  若95%区间完全位于有利方向而超出等效带，则标为**通过（改善）**，不称严格等效。
  **可确认回退**：不利方向超过5%且区间支持持续变化，继续定位；**不确定**：区间越界/过宽但不足以确认回退。
  超过5%的点估计或少量尾延迟必须显示，不能用“未触发回退”自动标等效。
- 如需更多样本，记录为独立确认阶段，重新冻结样本数；不反复追加直到出现有利显著性，不丢异常值。

## 发布

通过项以最新SHA、依赖、预热资格、配置和区间更新首页与PR；旧性能叙述移至`old/`，保留原始日志/数据和一个次要追溯入口。
仍有回退、未获资格或统计不确定的项保留在当前主表，不能隐藏。所有改动/失败用新label，独立研究分支维护工具或必要修复。

补充在正式统计前固定的实现细节：当前交付HTTP比较使用两侧匹配的`stack91`依赖
（ci8-clean已验证配置：NumPy1.26.4/Pandas2.2.3/Arrow25/Triton-Ascend3.2.2.post2），
旧历史Agents HTTP和norm比较仍使用原legacy依赖。`ir2`仅替换编译器的配置保留作诊断，不能混入主统计。
HTTP的R/Q均保留单字数字，输入长度与历史171-token模板一致；不同阶段使用互不重叠的R/Q组合。
早期pilot的三位R编号产生173-token输入，未计入正式验收。

在norm正式测试开始前补齐覆盖：增加独立的`agents-norm-short`比较（64输出token，其他条件与norm主比较相同，
同样8个AB/BA进程配对）。256-token长生成不能直接替代原64-token C4 TTFT场景的实测；两种长度分别报告。

norm入口尚未运行前与原`pipeline_case.py`再次逐字段核对：显式保留`prefix_cache_state_budget=0`、
`prefill_save_alignment=128`、`prefill_save_last_only=True`，避免依赖当前默认值；bench阶段原本不返回logprobs，
本轮也不返回，`logprobs_mode=raw_logprobs`配置保留。此修正不改变正在运行的HTTP入口。
