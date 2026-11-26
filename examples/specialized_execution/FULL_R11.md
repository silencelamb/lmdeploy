# Running the full GLM-5.3 R11 executor

`glm53_full_r11:FullR11` runs the entire GLM-5.3 target backbone under a single LMDeploy step plan. The program uses the R11 KDA, sparse attention, KPool, mHC, router, shared expert and routed EP8 implementations. The native expert grid has 132 CTAs. LMDeploy supplies the real model parameters, block table, paged caches and compact state slots. It also owns the head, sampler and scheduler. The 512-step device-controlled loop is an offline validation harness. [Executor](glm53_full_r11/__init__.py) · [Native ABI](glm53_full_r11/native.py) · [Expert weights and communication](glm53_full_r11/experts.py)

## Install the generated artifact

Use the PyTorch engine from this PR, Linux x86-64, Python 3.12, PyTorch 2.12.1+cu130, Triton 3.7.1 and TileLang 0.1.11. The published kit contains the compiled device programs, isolated R6 DeepGEMM package, math sources, native build sources/patches, pinned upstream revisions and license notices. It contains no model weights. The following commands use the immutable artifact commit in the contributor fork. [Artifact and checksums](https://github.com/silencelamb/lmdeploy/tree/5f27989ea6231edb44385d8e2b27037a4d669c8e)

```bash
export KIT=/path/to/new/r11-kit
mkdir -p "$KIT"
curl -fL https://raw.githubusercontent.com/silencelamb/lmdeploy/5f27989ea6231edb44385d8e2b27037a4d669c8e/glm53-r11-sm90a-cu130-cp312.tar.gz -o "$KIT/runtime.tar.gz"
(cd "$KIT" && echo '2d2545ce2a5488597af5dc534d9685dbaaf2e6dbd21a9ffe79b6fc1fc71618ac  runtime.tar.gz' | sha256sum -c - && tar -xzf runtime.tar.gz)
export PYTHONPATH="$PWD:$PWD/examples/specialized_execution:$KIT${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/path/to/GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
```

To rebuild the native libraries from the shipped sources run `bash "$KIT/sources/rebuild_r11.sh" /path/to/new/build-output` in the CUDA 13 development environment with C++17/20, `ptxas`, `nvcc` and libelf/libdw headers. Use the resulting output directory in `PYTHONPATH`. The builder can consume local, clean, pinned prerequisite checkouts via `R11_DEEPGEMM_SOURCE`, `R11_FLASHMLA_SOURCE` and `R11_CUTLASS_SOURCE`, and a non-system elfutils include directory via `R11_ELFUTILS_INCLUDE`. Starting from the original experiment tree, `python examples/specialized_execution/build_r11_bundle.py "$RESEARCH_ROOT" "$NEW_KIT"` assembles the prebuilt distribution and verifies the frozen Python sources.

## Launch serving

Set the Python path and artifact visibility in every worker environment. `LMDEPLOY_EP_CUDA_VISIBLE_DEVICES` is the ordered list of all eight local GPU ordinals (or visible GPU IDs mapped in the same order). Ray frontends keep their assigned GPU and expose that ordering to the actors for symmetric-memory rendezvous. The full program owns routed expert computation when selected. On main, the ordinary DeepEP fallback is already configured. The selectable SM90 MegaMoE backend in #5186 is optional.

```bash
export LMDEPLOY_EP_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LMDEPLOY_DP_MASTER_ADDR=127.0.0.1
export LMDEPLOY_DP_MASTER_PORT=24640
lmdeploy serve api_server "$MODEL" --backend pytorch --distributed-executor-backend ray \
  --language-model-only --trust-remote-code --tp 1 --dp 8 --ep 8 \
  --specialized-executor glm53_full_r11:FullR11 --max-batch-size 8 \
  --session-len 66560 --max-prefill-token-num 8192 --cache-block-seq-len 64 \
  --num-gpu-blocks 8321 --cudagraph-capture-batch-sizes 8 --model-name glm53 --server-port 24620
```

Admission requires the genuine 45-layer checkpoint geometry, BF16 activations and latent cache, FP32 block scales, indexed KDA state, the contiguous 36-expert shard belonging to each rank and full peer access. Each of the eight ranks must have eight real q=1 requests. The plan's host bound is 66,050, which admits the frozen driver's two extra metadata positions. Verified real lengths include 65,536 prefill plus 512 decode updates. The per-rank vote uses the existing CPU all-gather. A B7 rank, a serving dummy, a mixed phase or an unsupported length sends every rank to the appropriate ordinary path. The model's canonical weights and state remain available for that transition.

The generator finishes the expensive local packing before the engine sizes KV. It packs six experts at a time into the native K-major/SW128 layout and reserves 18 replica slots after the 36 local experts. The `prepare_distributed(ctx)` hook follows successful all-worker local preparation. It establishes symmetric buffers on the existing EP group and exchanges packed rows in fixed layer and peer order. Every rank uses the same planner and return protocol. Buffers are retired at the graph reset barrier. The compact KDA copy alone costs 2.125 GiB/rank. The full program additionally retains expert replicas, QKV and mHC layouts. Use free memory *after* preparation for cache sizing. Serving dummy ranks and unsupported shapes use the group-agreed ordinary path. Startup captures the exact distributed graph on all workers with reserved slots.

## Reproduce the checks

```bash
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_glm53.py \
  "$MODEL" --executor glm53_full_r11:FullR11 --context 65536 --prewarm --profile
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/benchmark_full_r11.py \
  "$MODEL" /path/to/new/results --route compare --kernel-trace
python examples/specialized_execution/analyze_full_r11.py --device /path/to/new/results
```

The 512-step driver reproduces the historical prompts, 64K real prefill per sequence in 8K chunks, 32 warmup steps, native BF16 full-vocabulary recording at every step, argmax-driven feedback, 45 model layers and the same CUDA 13 device-tail graph. Every case restores the same prefill state, requires finite logits and compares all tokens, every full-logit digest, four semantic states and five physical buffers/cursor. Exact numeric comparisons use SHA-256 of the tensor bytes. Graph dumps check the native calls on every rank. Three device-loop measurements and a host-queued replay control each record per-rank GPU timestamps. To obtain the critical-path mean, first take the maximum across ranks at each step.

For a run pinned to the historical trajectory add `--frozen "$R11/results/raw_recheck" --require-frozen-tokens`. The optional `--prefill-snapshot DIR` loads the actual historical prefill pools and first tokens after validating their schema and reporting the fresh-vs-archived comparison. A compatible snapshot is obtained by saving `dict(pools=initial, first_ids=first_tokens.cpu())` as `prefill-rank{rank}.pt` immediately after `initial_input` is built in the frozen `run_decode_profile.py`. The output records which of the two prefill conditions was used. Independent worker runs can write an oracle with `--route standard` and check it with `--route full --reference ORACLE_DIR`.

For a streaming benchmark use `serve_full_r11.py "$MODEL" OUTPUT --host ADVERTISED_IP --client-only` with the plain server. The client submits one short fallback request followed by 64 concurrent requests, eight per endpoint, via `/generate` using the exact original token IDs. It checks server-reported input/output lengths and stores token arrivals. To measure GPU boundaries select `profile_full_r11:ProfiledR11` on the server, set `GLM53_SERVING_TRACE` to a fresh directory shared by client and workers, and omit `--client-only`. The clean wave captures GPU start/end stamps for each real selected forward without CUPTI. A second wave collects a separate rank-zero Kineto trace. `profile_full_r11:RouteAuditR11` additionally snapshots physical expert counts for a diagnostic run. Drain requests before reading the buffers. [Trace provider](profile_full_r11.py) · [Streaming client](serve_full_r11.py) · [Results and scope](../../docs/zh_cn/advance/glm53_full_r11_results.md)

## Speculation

FullR11 declares an ordinary B8, Q1 plan. Its recurrent kernels write one output-state mapping per request, and its expert program budgets eight tokens per rank. The interface also accepts uniform MTP target verification, with the engine owning rejection and state commit. The working B2, Q4 provider reuses the stateless output projection from 34 layers. Extending the full program to B8, Q4 requires 32 token rows, four candidate states per request and larger or reused EP buffers. [MTP contract and verified scope](../../docs/zh_cn/advance/glm53_specialized_mtp_deployment.md)

The downloadable evidence includes per-rank timestamps, token and state comparisons, routing counts, client arrivals, the original and integrated DOT graphs, and compact CUPTI device timelines. [Download verification evidence](https://github.com/silencelamb/lmdeploy/blob/43228c050b6758a11cf5cfc5d97c7b1d8e789797/glm53-r11-h200-validation.tar.gz)
