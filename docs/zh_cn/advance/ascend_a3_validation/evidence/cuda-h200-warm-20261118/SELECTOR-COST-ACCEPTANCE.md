# Predeclared separate stable-selector cost experiment

- Two variants of the SAME upstream base: unchanged442097e8 versus existing isolated f06316de (only deterministic=True at CUDA NSA selector call). No migration source change is mixed into this contrast.
- Full GLM-5.3 checkpoint, TP8, original8H200, same CUDA dependencies/shared matched cache, BF16 activations/officialFP8weights, graph enabled, prefixoff,4276GPUblocks, identical32603-token prompt. No MTP in this option-cost contrast.
- Five fresh-process AB/BA pairs; each variant warms three times and measures five times for output budgets1 and32. Keep exact scores as observations; unordered default scores are NOT a numerical oracle for the stable option.
- Single-output-token prefill has identical fixed input and no divergent generated history fed back; report TTFT as full-model prefill cost. Do not call its1-token rate decode throughput.
- The32-output end-to-end cost is comparable only if actual generated token histories match across all measured samples; otherwise preserve raw timing as unqualified and rely on the fixed-input prefill result. No comparison of later logits after a first history divergence.
- Resource retries only for sampled foreignGPU PIDs, same predeclared rule as the migration rerun. No retry/selection on time or numeric results.
- This experiment measures the existing option, does not implement or merge a baseline repair, and is separate from the stable-vs-stable migration performance matrix.
