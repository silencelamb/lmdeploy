# MTP and Ray RPC investigation

Prepared 2026-11-16 Asia/Shanghai before new experiments.

- PR source is fixed at `5f1b1b3c12b1c198813f65275f778c7ad292fb02`; development and PR worktrees remain unchanged. Prior migration accuracy compares `fb5de87a` + dlinfer `3b909634` with native `6421cefe`; those versions are not silently replaced by the current head.
- First audit the **actual runtime** sampling route for Sage MTP3. Explicit `do_sample=False`, `top_k=1`, seed42; record the dispatch summary and verify every rejection output against an independent CPU longest-matching-prefix oracle. Fail immediately if random verification or multinomial sampling is reached. Do not infer deterministic model arithmetic solely from deterministic token selection.
- Record target argmax, runner-up margins, draft IDs, accepted/rejected counts and unchanged CPU/NPU RNG state at the verifier boundary. Logs are diagnostic; copying data to CPU can alter scheduling and these timings are not performance acceptance data.
- Ray: exercise a real loopback HTTP API server with Agents-A1, BF16, TP2, no MTP, graph decode, C1/C4, 256 output tokens. Follow each input-submission RPC from driver dispatch through worker entry/exit to driver completion. Record event-loop heartbeat delays to distinguish task/transport delays from blocked event loops.
- Keep production execution ordering and all original awaits. Instrumentation is validation-only. Do not remove synchronization or change algorithms to hide the effect.
- A3 observations do not certify H200 or establish an ARM-specific cause. CUDA hosts may share the Ray executor code, but need equivalent captures on the actual host. Small CPU-only Ray reproductions, if run, are a separate control.
- Preserve commands, source hashes, logs, failures and unverified limits. Do not resurrect the earlier unconfirmed HTTP C4 before/after regression; this is a fresh investigation of RPC timing on the HTTP path.
