# Agents gated RMSNorm performance investigation

Prepared 2026-11-16 Asia/Shanghai, before measurements.

- Production PR remains clean at `5f1b1b3c12b1c198813f65275f778c7ad292fb02`.
- Frozen pre-refactor **native** implementation: `342db85ecff3c0dc864a4bf8613c589c270f4141`. Neither side is the original LMDeploy+dlinfer baseline.
- Same Agents-A1 checkpoint, native environment without dlinfer, A3 devices 0/1, BF16, TP2, graph decode, no MTP/prefix cache, 256 output tokens, prefill 256, session 8192, batch limit 8, cache fraction 0.5.
- First isolate the changed norm: one engine and worker pair; switch all 30 norm modules between the frozen legacy module/kernel and the current public layer. Both objects share the identical loaded weight Parameter. Separate graph dictionaries and memory pools are retained for each mode. Warm both modes and both concurrency settings before measuring. This validation-only worker subclass does not modify the PR or its runtime environment.
- Eight paired measurements each at C1/C4, alternating AB/BA order, identical prompts within each pair. Report all samples, paired ratios, median throughput/TTFT/decode rate, variability and bootstrap intervals; do not select favorable runs. A throughput decrease >5% or TTFT increase >10% remains a confirmation trigger, not automatic proof of causality.
- Require exact output token IDs for every pair. Prior score-bit checks remain separately attributed; this streaming benchmark does not collect scores.
- After uninstrumented timing, record per-forward host enqueue duration and NPU event interval in separate diagnostic batches; verify decode graph capture/reuse and count prefill/decode work. Instrumented samples are excluded from end-to-end acceptance metrics.
- If these controls fail to explain the prior separate-process difference, investigate fresh alternating unmodified source-tree runs and/or profiler evidence. Do not call unresolved observations a pass or change model arithmetic to improve a benchmark.
- Preserve all commands, environment/source records, raw samples, failed attempts, and analysis on the fork report branch, outside the code PR.

## Follow-up protocol recorded before GC observation

The first same-engine substitution experiment completed its timings but failed its final all-pairs assertion: the first graph-capture warmups differed from token index 1, while every subsequent measurement and diagnostic pair matched exactly. Preserve that run as a failed harness check; it is not a full-model correctness pass. For subsequent diagnostic experiments, keep first captures in a separate `capture_warmup` phase and report their token comparison explicitly. Then run a second warmup and require exact token equality in every steady warmup, measurement and diagnostic pair. The substitution experiment only supports warmed performance conclusions; original unmodified-source correctness traces remain the accuracy evidence.

Add lightweight GC callbacks in the driver and both workers, recording monotonic timestamps, generation, duration and process ID. Correlate >80 ms decode gaps after the first output token with these events. If events explain the pauses, perform a validation-only GC-disable control, without changing the production GC policy. All samples and policies must be disclosed.
