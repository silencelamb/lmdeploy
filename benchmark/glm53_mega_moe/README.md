# GLM-5.3-Flash SM90 MegaMoE

The [H200 measurements](../../docs/zh_cn/benchmark/glm53_sm90_megamoe.md) include fixed-I/O, AgentX, memory and numerical results. The [machine-readable tables](results/h200_megamoe.json) retain the per-arm settings and counts.

The PyTorch engine accepts `--moe-backend default`, `deepgemm_mega_moe` or `flashinfer_mega_moe`.
Both MegaMoE implementations require one SM90 node, DP8 + EP8 + TP1, BF16 activations
and a 128×128 E4M3 block-FP8 checkpoint with FP32 dequantization scales.
The backend choice applies to the main model. By default the launcher uses no speculative model.

EP ranks use the scheduler's DP-wide decode flag to choose the collective. By
default, a step with any prefill rank uses DeepEP normal dispatch. Decode uses
one symmetric workspace per model and rank. CUDA Graph padding and dummy decode tokens
carry invalid expert IDs. The model keeps its FP32 noaux router, its routed
weight multiplier of 2.5, and its shared expert.
The DP launcher preserves the caller's GPU order for MegaMoE. Each API
process reserves its own GPU in Ray. The worker uses the same peer-visible
CUDA ordinals as the other EP ranks for symmetric memory.
Set `LMDEPLOY_GLM53_DEEPGEMM_FUSE_SHARED=1` before startup to exercise
DeepGEMM's fused shared expert. In this mode the input staging kernel writes
both activation layouts from the same per-token quantization. The native
prefill path keeps its separate shared MLP. FlashInfer combines block
quantization, route staging and tail masking in one Triton launch.

The default `dual` weight layout keeps both FC1 layouts. An extra FC1
costs 23.625 GiB per rank for 42 layers with 36 local experts. The two
other layouts load each checkpoint shard directly into the 8-row gate/up
interleave. They preserve the original E4M3 bytes and FP32 scales.

| Setting | Prefill FC1 execution | Extra FC1 payload per rank |
| --- | --- | ---: |
| `dual` | Native grouped GEMM reads the canonical copy | 23.625 GiB |
| `compact` | Unpack the experts addressed by dispatch into a model-wide CUDA buffer | 576 MiB |
| `interleaved` | Extended grouped GEMM reads the decode layout directly. SwiGLU indexes the interleaved BF16 output. | 0 bytes |
| `interleaved_fused` | The same grouped GEMM. SwiGLU also quantizes FC2 inputs and emits TMA-aligned scales. | 0 bytes |

Set `LMDEPLOY_GLM53_DEEPGEMM_WEIGHT_LAYOUT` or
`LMDEPLOY_GLM53_FLASHINFER_WEIGHT_LAYOUT` before startup. The optional
`interleaved` and `interleaved_fused` layouts require
[the DeepGEMM extension](deepgemm-interleaved-fc1.patch).
FlashInfer 0.7.4 uses one scale per 128 physical rows of FC1. Its stock
path requantizes the half with the smaller scale. To preserve every checkpoint
scale and use a compact layout, apply [the split-scale extension](flashinfer-sm90-split-scales.patch)
and set `LMDEPLOY_GLM53_FLASHINFER_EXACT_SCALES=1`.
The patched kernel loads both scales from the same block index and chooses
one for each gate/up register group during WGMMA accumulation. Both backends
reuse the FC2 payload. FlashInfer accepts optional tuning parameters through
`LMDEPLOY_GLM53_FLASHINFER_KNOBS` as a JSON object.

## Environment

- LMDeploy baseline `63549b282790f77ebf4bf2a0d6e44a70de14d85b` with PyTorch
  `2.12.1+cu130`, CUDA 13.0 and LMDeploy's DeepEP build.
- DeepGEMM commit `44cba7d`. Use its pinned CUTLASS and DeepJIT submodules.
  Build with PyTorch 2.14 headers and `libdw-dev` / `libelf-dev`. The generated
  extension uses the stable PyTorch ABI. With DeepGEMM 2.5.0 in the serving
  image, use `install_deepgemm_provider.py WHEEL PREFIX` and put `PREFIX`
  on the Python path. This preserves DeepGEMM 2.5.0 for native KPool and
  DeepEP GEMM calls in all configurations.
- FlashInfer `0.7.4` (`77039051b0be921788dc2784ec41423b558878ec`),
  CuTeDSL `nvidia-cutlass-dsl[cu13]==4.8.0`, and `nvshmem4py-cu13`.
  Use the same CUDA / NVSHMEM major versions as the engine. Keep the
  CuTeDSL and DeepGEMM JIT caches between repeated launches.

Build the interleaved prefill provider from the pinned DeepGEMM checkout
after initializing its submodules. In the serving image, expose the PyTorch
2.14 headers ahead of the image's 2.12 headers.

```bash
git apply "$LMDEPLOY_SRC/benchmark/glm53_mega_moe/deepgemm-interleaved-fc1.patch"
CXXFLAGS="-I$TORCH214_INCLUDE" DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
  python setup.py bdist_wheel --dist-dir "$OUT"
python "$LMDEPLOY_SRC/benchmark/glm53_mega_moe/install_deepgemm_provider.py" \
  "$OUT/deep_gemm-2.8.1-cp310-abi3-linux_x86_64.whl" "$PROVIDER_PREFIX"
```

## Serving and Fixed I/O

Run the server for one backend at a time. It exposes eight worker ports and eight
monitoring ports. Set `GLM53_PORT` and `GLM53_MONITORING_PORT` as needed.

```bash
bash benchmark/glm53_mega_moe/serve.sh "$MODEL_SNAPSHOT" "$MOE_BACKEND"
python benchmark/glm53_mega_moe/fixed_io.py \
  --urls "${WORKER_URLS[@]}" --tokenizer "$MODEL_SNAPSHOT" --output "$RUN/fixed"
```

The client sends input IDs and requires exactly 1,024 streamed IDs for every
request. It warms each shape with new prefixes and balances requests over
the eight workers. The JSON summaries use per-request TTFT and TPOT, and
per-token ITL from the arrival timestamps. Raw streams and GPU memory samples
sit beside every summary. Use a fresh output directory for each run.
Pool the request-level samples within each run and compare runs with
`python benchmark/glm53_mega_moe/summarize.py "$RUN1" "$RUN2" "$RUN3"`.

`run_serving.sh` automates server and router lifecycle in the benchmark
container. It expects the LMDeploy source at `/src`, the model repository
at `/model`, and `MEGA_MOE_ARTIFACTS` mounted at `/artifacts`. Set
`MODEL_SNAPSHOT` to the host snapshot directory. The launch records every
command, resident HBM and GPU telemetry. `AGENTX_CONCURRENCIES=""` runs only
Fixed I/O. `SKIP_FIXED_IO=1` runs only AgentX.

```bash
AGENTX_CONCURRENCIES="8 32" bash benchmark/glm53_mega_moe/run_serving.sh \
  "$MOE_BACKEND" "$KV_BLOCKS_PER_RANK" "$RUN_NAME"
```

Use these environments to reproduce the interleaved serving configurations.

```bash
# DeepGEMM
export LMDEPLOY_GLM53_DEEPGEMM_FUSE_SHARED=1
export LMDEPLOY_GLM53_DEEPGEMM_WEIGHT_LAYOUT=interleaved
export MEGA_MOE_DEEPGEMM_PATH=/artifacts/deps/deepgemm-interleaved

# FlashInfer
export LMDEPLOY_GLM53_FLASHINFER_EXACT_SCALES=1
export LMDEPLOY_GLM53_FLASHINFER_WEIGHT_LAYOUT=interleaved
export MEGA_MOE_FLASHINFER_PATH=/artifacts/build/flashinfer-split-scales
```

For equal KV capacity, pass the same `--num-gpu-blocks` to all three commands and
verify usable blocks in the server logs. For a matched HBM budget, choose
blocks per backend from the measured resident bytes and the per-block size.
Check idle memory and the per-GPU peak on the same workloads. LMDeploy's
`--cache-max-entry-count` is a fraction of remaining free memory after weights
and runtime reservations, so a common fraction yields different total
allocations.

### FlashKDA prefill on H200

The default KDA prefill uses FLA. Set `LMDEPLOY_GLM53_FLASHKDA_LIBRARY` to
the worker-visible path of the FlashKDA shared object to enable its indexed
prefill adapter. The benchmark used the stable-ABI extension from vLLM
commit `5328ae2fef`, built against FlashKDA `9d9e928e`. Its SHA256 is
`108f236bf5732ab1a643cb2ede54003d0d8a282c89e0f9a3c6b355daa23db250`.

```bash
unzip -j "$VLLM_WHEEL" 'vllm/_flashkda_C.abi3.so' -d "$PROVIDER_PREFIX"
export LMDEPLOY_GLM53_FLASHKDA_LIBRARY="$PROVIDER_PREFIX/_flashkda_C.abi3.so"
python benchmark/glm53_mega_moe/bench_kda_prefill.py \
  --library "$LMDEPLOY_GLM53_FLASHKDA_LIBRARY" --output kda-prefill.json
```

The adapter passes row-strided Q/K/V directly from the convolution. It
snapshots indexed inputs before running the fused recurrence, then exports
final states and interior prefix checkpoints to the scheduler's reserved
units. A terminal checkpoint uses the final state, including partial tiles.
The measured path uses BF16 activations, 128 channels per head and FP32
recurrent state. FlashKDA changes prefill rounding. For comparisons, hold
the MoE backend, prefill capacity, MTP depth and KV blocks fixed.

## AgentX

Start the same sticky-least-loaded router for each backend and point the
client at its HTTP port. Use AIPerf with the InferenceX AgentX MVP scenario and the Weka 256K trace cache.
Set `AIPERF_BIN` or put `aiperf` on PATH. Configure `HF_HOME` for the staged
public dataset and tokenizer. `AGENTX_SEED` selects the client seed.

`GLM53_SPEC_TOKENS=3` or `5` enables the model's checkpoint MTP head with the
`deepseek_mtp` proposer. The default is zero. The selected MoE backend applies
to target verification. The draft head keeps native DeepEP. Repeat the idle
HBM calibration after changing the draft length. Target and draft KV each
occupy a physical block, and the target's compact KDA bank reserves one
destination per verified token at each request slot.

```bash
lmdeploy-router --host 0.0.0.0 --port 20970 --policy sticky_least_loaded \
  --worker-urls "${WORKER_URLS[@]}" --intra-node-data-parallel-size 1 \
  --request-timeout-secs 18000 --health-check-interval-secs 5
```

Wait for the router's `/health` to succeed after each worker restart.

```bash
for concurrency in 8 32; do
  bash benchmark/glm53_mega_moe/agentx.sh "$ROUTER_URL" "$concurrency" \
    "$RUN/agentx-c$concurrency" --server-metrics "${METRICS_URLS[@]}"
done
```

The thresholds are TTFT p50 < 3 s and TPOT p50 < 25 ms. Per-request goodput
uses 3,000 ms TTFT and 40 output tokens/s per user, matching the earlier
AgentX setup. The defaults are a 1,800-second profile and ten
warmup requests per lane. Shorter profiles use the diagnostic override. Keep `command.sh`,
`profile_export.jsonl`, the server metrics and the export summary.

## Layer and fixed shape

Use an exclusive 8-GPU process group and a different output directory for each
backend. The layer harness loads an entire FP8 MoE layer and its router and
shared expert from the checkpoint. It uses real weights with seeded inputs and
compares normal, concentrated and masked routes to DeepEP. The optional traces
record the routed kernel(s), staging, and separate shared-expert execution.
Batch 8 also replays a single graph with changing valid row counts, then
executes with zero physical rows on half of the ranks.

```bash
torchrun --nproc-per-node=8 --master-port=20898 \
  benchmark/glm53_mega_moe/verify_layer.py --model "$MODEL_SNAPSHOT" \
  --backend "$MOE_BACKEND" --profile --output "$RUN/layer"
```

The whole-decode harness creates eight 65,536-token real prefills on each rank,
captures a decode graph, and restores the same KV and compact-state pools for
each timed repeat. Run a default reference and pass its output directory with
`--reference` for MegaMoE. Candidate accuracy uses the default's input token
trajectory while recording the candidate's own predictions.

```bash
torchrun --nproc-per-node=8 --master-port=20899 \
  benchmark/glm53_mega_moe/fixed_shape.py --model "$MODEL_SNAPSHOT" \
  --backend default --profile-steps 4 --output "$RUN/reference"
torchrun --nproc-per-node=8 --master-port=20899 \
  benchmark/glm53_mega_moe/fixed_shape.py --model "$MODEL_SNAPSHOT" \
  --backend "$MOE_BACKEND" --profile-steps 4 --reference "$RUN/reference" \
  --output "$RUN/$MOE_BACKEND"
```

Per-rank outputs include GPU and wall times, CUDA Graph count, full-vocabulary
logits at selected steps, and samples from the KV, KDA and KPool buffers.
`prefill_kv` and `decode_kv` report numeric MLA latents and dequantized KPool keys
with their scales. Raw allocation bytes are retained for exact storage comparisons.
Use `--eager` to exercise the same inputs without graphs. A serving run also
exercises the scheduler and naturally varying batches. The harness checks sampled numerical values for finiteness. With a reference,
`--check-prefill` also enforces identical prefill state samples and first tokens. The optional
`--record-moe-layers 3 22 41` stores actual decode MoE inputs, which
`verify_layer.py --layer 22 --activations "$RUN/reference"` can replay
with the matching checkpoint layer. Set `--capacity 32 --batches 8` for
an 8-row replay using the production workspace geometry. Aggregate the whole-step timings and
rank-0 kernel durations with `summarize_shape.py`.

## vLLM PR comparison

The fixed I/O client can use consecutive random vocabulary IDs with rolling
concurrency. Supply `--case INPUT:OUTPUT:CONCURRENCY:REQUESTS` for each
workload and use the same seed and placement policy for paired servers.
`--placement least-loaded` distributes requests using the current number of
in-flight requests on each worker. The default placement is round robin.

```bash
python benchmark/glm53_mega_moe/fixed_io.py \
  --urls "${WORKER_URLS[@]}" --tokenizer "$MODEL_SNAPSHOT" --output "$RUN/fixed" \
  --random-tokens --seed 0 --repeats 1 --case 1024:1024:64:256 --case 8192:256:64:256
```

Set `LMDEPLOY_GLM53_MEGA_PREFILL_CAPACITY=8192` to enable MegaMoE during
prefill and mixed DP steps. All ranks select the same workspace from the
global maximum token count. DeepGEMM selects its math tile using the global
mean. FlashInfer keeps smaller workspaces for short cache-hit turns. The
native prefill path remains available when a step exceeds the configured
MegaMoE capacity. The default capacity of zero uses the native prefill path.
The fused prefill changes FP8 rounding and route reduction order. Check
free generation on the target workload when enabling it, especially with
long contexts and recurrent state.

`bench_prefill.py` measures the FP8 expert layer using the same checkpoint,
both with equal local prefill lengths and with a mix of prefill, decode and
dummy ranks. The runner also records `GLM53_MAX_PREFILL_TOKENS`,
`GLM53_MAX_BATCH_SIZE`, and optional `GLM53_PCG_TOKENS`. Piecewise graph
plans use exact small token counts on mixed-step decode ranks and run the
expert collective using live DP metadata.

## Upstream implementation references

- [DeepGEMM's SM90 block-FP8 MegaMoE](https://github.com/vllm-project/DeepGEMM/pull/57)
- [The GLM-5.3 adapter in vLLM](https://github.com/vllm-project/vllm/pull/66812)
- [The FlashInfer SM90 FP8 integration in vLLM](https://github.com/vllm-project/vllm/pull/62603)
- [FlashInfer SM90 tuning](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/moe_ep/shims/sm90/cutedsl_megamoe/TUNING.md)

Set `LMDEPLOY_DP_SYNC_METADATA=1` to measure synchronous completion of the CPU DP metadata collective in low-concurrency decode. Measure the higher-concurrency tradeoff before enabling it. See the [execution results](../../docs/zh_cn/benchmark/glm53_tpot.md).

### Measuring prefill interference with MTP

With eight worker URLs in DP-rank order, use the following client for each
MTP3 or MTP5 server. Pass the same `--draft-depth` as the server. It runs two cold 8K/256 rolling loads at C=64 and measures the
same 32 warm probes with and without a new wave of 24 long prefills. The
second repetition reverses the arm order. Each probe emits exactly 512 tokens.

```bash
python benchmark/glm53_mega_moe/mtp_interference.py \
  --draft-depth 3 --api generate --urls "${LMDEPLOY_URLS[@]}" --monitor-port 20980 \
  --tokenizer "$MODEL_SNAPSHOT" --output "$LMD_RUN/fixed"
python benchmark/glm53_mega_moe/mtp_interference.py \
  --draft-depth 3 --api completions --urls "${VLLM_URLS[@]}" \
  --tokenizer "$MODEL_SNAPSHOT" --output "$VLLM_RUN/fixed"
```

The measured runs use the lifecycle runner's `exit_code` file. After checking
both runs completed successfully, summarize their per-request latency pairs,
cache/acceptance counters and streaming gaps with
`python benchmark/glm53_mega_moe/summarize_mtp_interference.py "$LMD_RUN" "$VLLM_RUN"`.
