## Motivation

Ascend A3 inference currently depends on dlinfer installing LMDeploy framework patches at import time. Move that implementation into LMDeploy's PyTorch engine so `device_type="ascend"` selects an explicit native backend and the inference environment can run without installing or importing dlinfer.

## Modification

- Add `backends/ascend` and `kernels/ascend`, reusing the existing `OpsBackend`, `BuildSpec`, and `GraphRunner` interfaces. Preserve the migrated operator calculations and source attribution.
- Replace LMDeploy model/configuration monkey patches with explicit platform dispatch for graph buffers, GDN/MTP state, MoE communication, ModelSlim/AWQ loading, sampling, and shared vision helpers. Ascend uses `torch.npu.NPUGraph`; CUDA retains its own implementations.
- Give gated RMSNorm the same structure as ordinary RMSNorm: `nn.RMSNormGated` owns checkpoint weights, a typed BuildSpec selects stateless CUDA/Ascend implementations, and all three direct model callers use it. Remove the special factory and `default/gated_norm.py`; retain CUDA's original FLA 0.5.2 calculation and fix Ascend's previously ignored sigmoid/dtype arguments.
- Include the callable-import fix found during A3 validation, canonical `ascend` platform naming, native runtime/package dependencies, and an A3 source-build Dockerfile and installation instructions.
- Add migration contracts and port the existing Ascend operator tests. Common `pytest tests` collection excludes Ascend hardware tests before importing them; an A3 job must opt in with `--run-ascend-hardware`. CPU contracts remain part of common CI.

This PR contains only the native migration. Mooncake [#5030](https://github.com/InternLM/lmdeploy/pull/5030), GLM W8A8 optimization [#5001](https://github.com/InternLM/lmdeploy/pull/5001) and its dlinfer counterpart, the separate gated-RMSNorm performance optimization, and decode megakernels remain separate follow-ups. The dlinfer cleanup is a separate repository change and is not required to install this native backend.

## Compatibility

Keep `PytorchEngineConfig(device_type="ascend")` / `--device ascend`. MACA/CAMB retain their existing dlinfer path. Native Ascend still requires torch_npu, CANN/NNAL/ATB, Triton Ascend, and external `triton-ascend-kernels`; it also retains torch_npu's explicit `transfer_to_npu` initialization. Removing the dlinfer dependency does not remove those vendor dependencies or claim that their compatibility shim has no process-wide effects.

The migration targets A3. CUDA, A2/300I Duo, MACA, and CAMB hardware behavior has not been certified by the A3 runs below.

## Validation

The complete [validation overview](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/README.md), [model/operator comparison and reproduction evidence](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/migration_validation.md), and [PR preparation checks](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/native_pr_20261115.md) live on a separate branch of my fork. The comparison reports and archived data are excluded from this PR. The [gated RMSNorm refactor report](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/gated_norm_refactor_20261115.md) adds provenance, complete caller coverage, actual checkpoint weights, output traces, and performance confirmation.

### Fixed comparison versions

| Role | Commit |
| --- | --- |
| Baseline LMDeploy | [fb5de87a](https://github.com/InternLM/lmdeploy/commit/fb5de87a45df9999a19a569ed9441df8edfa6975) |
| Baseline dlinfer | [3b909634](https://github.com/DeepLink-org/dlinfer/commit/3b9096341a41e9d067104a39d914e47ca2d7ec50) |
| Native LMDeploy used for full A3 model comparisons | [6421cefe](https://github.com/silencelamb/lmdeploy/commit/6421cefea94edd350e419f556158639556357a27) |

The PR is prepared on upstream [442097e8](https://github.com/InternLM/lmdeploy/commit/442097e83799c2e85df9d6bbce35c7c79ea05dfe), incorporating subsequent test isolation, platform-name cleanup, documentation, and repository formatting. Initial preparation checks below run on the prepared PR tree. Historical full-model/performance measurements remain attributed to their actual tested commit. Additional gated-norm refactor comparisons on the prepared PR tree are listed separately below.

### Results

The controlled comparison used isolated baseline/native environments, identical checkpoint files and configurations, and an import guard forbidding dlinfer in native parent and worker processes. Main models were Sage-A1.5 and Agents-A1 (BF16, TP2), with Qwen3-8B (TP1) as a dense supplement. The A3 image used Python 3.12.13, torch 2.10.0, torch_npu 2.10.0.post4, and CANN 9.1.0.

| Coverage | Observed result |
| --- | --- |
| Sage-A1.5, tested matching non-MTP configurations | **Bitwise-identical recorded outputs.** Eager and graph/prefill256 each matched 21/21 sequences, 1344 generated tokens, and all 6720 collected top-k logprob values; graph/prefix and fixed-history scoring also matched exactly |
| Agents-A1, tested matching non-MTP configurations | **Bitwise-identical recorded outputs.** Eager matched 21/21 sequences, 1344 generated tokens, and all 6720 collected top-k logprob values; graph/prefix, fixed-history scoring, and the long-context case also matched exactly |
| Sage MTP3 targeted prefix/quality cases | 16/16 responses, 56 generated tokens, and 280 collected scores matched bitwise; the long-context case also matched all 8 output tokens and 40 collected scores. Long-generation repeatability is described separately below |
| Qwen3-8B dense supplement | Exact before/after generated-token and collected top-k score agreement in the tested matching configurations |
| Prefix cache and long context | Real prefix-cache hits verified; both main models completed the 261845-input-token retrieval case correctly |
| Native operator suite on the historical tested commit | 26 passed, 3 failed; the 3 MLA failures also occur on baseline. The passing count includes 16 CPU convolution-ring reference cases, not 26 NPU kernels |
| Initial prepared-PR CPU/shared-interface checks without torch_npu | 118 passed, 10 CUDA-unavailable failures, 1 skipped; upstream control: 86 passed with the identical 10 failure IDs/messages. The added Ascend CPU contracts account for 32 passes; the skipped module requires torch_npu |
| Initial prepared-PR contracts with the A3 vendor runtime | 34 passed |
| Initial prepared-PR real NPU cache-copy test with hardware opt-in | 1 passed |
| Applicable repository pre-commit hooks and diff checks | Passed on changed files; no C/C++/CUDA source files changed |

The bitwise result covers the recorded token IDs, complete top-k token-ID sets, and returned logprob values. A fresh [audit of the saved traces](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/evidence/native-pr-20261115/results/recorded-output-bitwise-audit.json) compared every saved score's binary representation with zero tolerance; full internal tensors were not recorded.

### Standard gated RMSNorm refactor ([5f1b1b3c](https://github.com/silencelamb/lmdeploy/commit/5f1b1b3c12b1c198813f65275f778c7ad292fb02))

Compare the frozen pre-refactor native PR [342db85e](https://github.com/silencelamb/lmdeploy/commit/342db85ecff3c0dc864a4bf8613c589c270f4141) against the refactored tree using the same A3 dependencies and a dlinfer import guard.

- **98 CPU/interface/model-loading checks passed** on each of the PR and development trees; **61 skipped**, including 56 real CUDA checks that cannot execute on A3.
- **55 real A3 arithmetic/NPUGraph checks passed**. **32 frozen-kernel/Module comparisons were bitwise equal** across dtypes, shapes and legacy kernel options.
- **208 actual checkpoint norm scales passed**: 174 SiLU layers were bitwise equal to the old kernel; GLM-5.3-Flash's 34 sigmoid layers passed an independent mathematical reference. The old Ascend factory had ignored its explicit sigmoid argument.
- **Sage-A1.5, Agents-A1, dense Qwen3.5, Intern-S2 Preview and Intern-S2 Mobius** each matched all 21 trace responses, 1344 generated tokens and 6720 recorded scores bitwise. Sage/Agents were rechecked on final commit [5f1b1b3c](https://github.com/silencelamb/lmdeploy/commit/5f1b1b3c12b1c198813f65275f778c7ad292fb02); source snapshots for the other runs are archived. Intern-S2 required loading RDKit before torch_npu in both parent and worker processes in this image.
- All direct callers and derived classes are covered at construction/loading level, including Qwen3-Next, Qwen3.5 dense/MoE, InternDecision, Intern-S2 Preview/Mobius, MetaMoe and GLM5-Next. Full A3 Qwen3-Next and InternDecision attempts encounter the same configuration/ordinary-RMSNorm errors on both native pre-/post-refactor trees. GLM5-Next also lacks an Ascend KDA implementation. This comparison does not establish their status on the older LMDeploy + dlinfer baseline. These are **not full-model passes**.
- **Performance: no confirmed gated-norm regression; intermittent host waits remain open.** The historical five-repeat, 256-token Agents comparison recorded C1/C4 throughput changes of −1.38%/−5.15% and TTFT changes of +14.15%/−6.92%. A [follow-up investigation](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/gated_norm_performance_20261116.md) found the actually compiled old/new norm NPU binaries **byte-identical**. Eight AB/BA pairs per concurrency, sharing the same workers and checkpoint weights after warmup, did not reproduce a stable decrease: C1/C4 median throughput changed by −1.40%/+10.88%, with wide paired confidence intervals; this is not an acceleration claim. The follow-up localized reproduced 245–551 ms streaming gaps mainly to **222–533 ms waiting for previous Ray worker-submission RPCs** in `RayExecutor.forward_async`, shared by both refactor revisions. The Ray-internal trigger and attribution of every historical sample remain unresolved. The report preserves all samples and the dynamic-substitution experiment's first-capture output mismatch; that experiment supports warmed performance diagnosis, not a new cold-start correctness pass.

Full details and raw evidence are in the [dedicated report](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/gated_norm_refactor_20261115.md). The [A100 CI job](https://github.com/InternLM/lmdeploy/actions/runs/43727003277/job/132851013103) subsequently passed **all 83 checks in the new gated-norm test file, including 56 real CUDA/graph checks**, plus all eight model construction/loading checks. The complete unit job has 6424 passed, 3 failed and 400 skipped; the three failures involve test fixtures missing `BaseModelAgent.cache_config`. Overall GitHub CI is not all green. The ETE workflow passed; complete GPU checkpoint comparisons remain pending. See the [follow-up report](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/gated_norm_performance_20261116.md) for evidence and scope.

### Known observations and limits

- **MTP non-determinism already exists in the baseline.** Both versions exhibit this behavior under fixed-seed repeats: the baseline matched 14/21 sequences against itself, and native matched 13/21 against itself; the original baseline/native MTP3 long-generation comparison matched 13/21. This is retained as a non-blocking observation also present in the baseline, not a confirmed newly introduced migration regression. These long-generation cases are documented separately from the bitwise-passed cases above.
- **Shared baseline failures remain:** three paged/decode MLA cases fail with the same CANN keyRope-contiguity error, and the MTP zero-output-token input-scoring path fails in both versions.
- **Environment limitations:** the supplied image exceeds some declared Python/torch dependency bounds and `pip check` is not clean. The external Triton kernel snapshot is pinned in the report, including its pre-existing patches. These runs are a controlled migration comparison, not a clean supported-wheel installation certification.
- **Not covered:** complete NVIDIA GPU model comparisons and TurboMind execution, complete GLM/DeepSeek checkpoints and quantized-model paths, multimodal inference, DP/EP and multi-node configurations, production-scale stress/soak tests, and standard benchmark accuracy such as GPQA. Small operator and CPU contract tests do not substitute for these checks.

Mooncake's separate integration results are documented in the linked report but are not counted as validation of functionality delivered by this PR.

### Planned NVIDIA GPU regression (pending)

The author will run the following CUDA comparisons on GPU. The acceptance target for **every model** is bitwise agreement between the unmodified upstream CUDA baseline and this PR's CUDA path, using the same GPU, checkpoint, software stack, dtype, TP layout, prompts, sampling, and cache configuration. Record both commit IDs; the current pair is base [442097e8](https://github.com/InternLM/lmdeploy/commit/442097e83799c2e85df9d6bbce35c7c79ea05dfe) and PR [5f1b1b3c](https://github.com/silencelamb/lmdeploy/commit/5f1b1b3c12b1c198813f65275f778c7ad292fb02).

| Model | Acceptance target | Status |
| --- | --- | --- |
| GLM-5.3-Flash | Bitwise agreement | Pending GPU validation |
| InternDecision | Bitwise agreement | Pending GPU validation |
| Sage-A1.5 | Bitwise agreement | Pending GPU validation |
| Agents-A1 | Bitwise agreement | Pending GPU validation |
| Intern-S2 | Bitwise agreement | Pending GPU validation |
| GLM-5.3 | Bitwise agreement | Pending GPU validation |

Require 100% generated-token-ID agreement and byte-identical recorded numerical outputs with the same dtype and zero tolerance. Compare scores under fixed token histories, check eager/graph configurations separately, and establish non-MTP results first. Test MTP separately where supported, including baseline self-repeat controls; any non-repeatable case remains separate from the bitwise-pass count. These are migration-before/after checks on GPU, not GPU-versus-A3 comparisons.

## Checklist

- [x] Applicable lint hooks, Ascend contracts, and the NPU cache-copy check passed; expanded-suite CUDA limitations disclosed.
- [x] Installation and test-selection documentation updated.
- [x] Fixed baseline/native revisions, failures, and unverified coverage disclosed.
- [x] Comparison reports kept on the fork documentation branch.
- [ ] Complete the six-model NVIDIA GPU bitwise-regression matrix above.
