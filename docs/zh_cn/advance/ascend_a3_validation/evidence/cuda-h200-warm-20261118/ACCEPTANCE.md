# H200 fresh warmed performance campaign

- Started 2026-11-17; base 442097e8 and PR f64ab506 remain fixed. Current PR HEAD 2ccf38a4 has identical runtime/test/dependency trees. No main update.
- Same preserved CUDA dependencies, exact checkpoints, original GPU IDs, BF16 activations/FP8 weights where supplied, TP and cache block counts. Same original C1/C4 prompt sequences and 256 output tokens; graph enabled, prefix disabled.
- Repeat all previous H200 operator, pipeline and HTTP performance scopes, non-MTP before MTP3. Full large models use GPUs 0-7; Sage/Agents/Decision use their original single GPU. GLM full-model runs use both isolated deterministic-selector controls and are labelled separately.
- Copy the preserved matched Triton cache into a campaign-only shared directory. Preserve initial and final cache hashes; let the unchanged tuner populate any genuinely missing specializations. No manually selected winner.
- Warm each measured C1/C4 workload at least three times in every fresh process; Sage/Agents HTTP also warm exact 668-token MoE hint2 using the existing raw-completion fixture. Keep all warmup outputs/logs.
- At least five fresh-process AB/BA pairs per scope/mode. Five measured batches per concurrency per process, summarized per process before paired statistics; do not treat repetitions as independent processes.
- Original trigger retained: adverse paired median >5%, at least 4/5 pairs >5%, paired bootstrap95% interval excluding zero. This is not a claim of ±5% equivalence. Additional localization required for any persistent new signal.
- Preserve output token IDs, input/output lengths, cache hits, per-request TTFT/TPOT, batch TPS, timestamped 200ms NVML memory/clocks and import audits. Separate nontimed exact-score probes from timed generation.
- Previously unqualified HTTP C4 is rerun, not automatically certified by warmup; require matched workload and numerical evidence before removing that qualification. Preserve original data and any new failures.
- No algorithm, collective precision/order, scheduler, affinity, synchronization or acceptance-threshold changes. No NPU packages imported. No A3 rerun, and no renewed investigation of the closed cross-prefix-cache observation.
- Update original PR body and fork docs branch only after reviewing results; never merge PR.

## Clean restart 2026-11-18

The user coordinates an exclusive 8-GPU window. Every process acquires all eight task locks and checks all eight GPUs idle before launch; a separate read-only GPU PID sampler records overlap. Restart uses a fresh copy of the original matched cache, not the cache modified during contaminated trials. HTTP exact-ID/top5 probes are outside timing; the identical original AsyncEngine.generate method is restored before warmups and timed calls, with explicit control records. Full C4 prefill shape is warmed for every generative model.
