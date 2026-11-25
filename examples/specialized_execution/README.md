# Delivering a generated decode executor to LMDeploy

A provider is an importable Python class plus its optional native artifacts. The worker calls it for one target-model forward. This example runs the GLM-5.3 backbone with the R11 compact BF16 output projection in its 34 KDA layers. Its public configuration is `specialized_executor="glm53_r11:R11Projection"`. The CUDA implementation is confined to this directory. [Python API](../../lmdeploy/pytorch/specialized_execution.py) · [Working provider](glm53_r11.py) · [Design and MTP analysis](../../docs/zh_cn/advance/specialized_execution.md) · [R11 source audit](../../docs/zh_cn/advance/glm53_monokernel_audit.md)

The full `glm53_full_r11:FullR11` executor also owns mHC, KDA, sparse MLA/KPool, shared experts and the 132-CTA EP8 program. Its portable runtime, source build, 512-step checks and real streaming benchmark have a dedicated guide. [Full R11](FULL_R11.md)

## Generator handoff

Ship the class, a stable `artifact_id` covering the generated code, weight layout and collective protocol, the supported runtime/toolchain versions, a reproducible native build, and an oracle against the standard model. All workers must install the same build. A provider names each distinct graph/collective program with a unique plan. A registry or changes to the server API are unnecessary for additional packages.

The smallest runnable adapter below executes the standard forward on single-worker B1 decode. Save it as `my_generated.py` in an importable directory and select `my_generated:Example`. A generated implementation replaces `run` and prepares its weights in `__init__`. The bundled H200 provider shows a native implementation of that substitution.

```python
from lmdeploy.pytorch.specialized_execution import DecodePlan, SpecializedExecutor

class Example(SpecializedExecutor):
    api_version = 1
    artifact_id = "reference-forward-v1"

    @classmethod
    def describe(cls, ctx):
        if ctx.dist.dist_config.world_size == 1:
            return (DecodePlan("ar-b1", batch_size=1, graph_safe=True),)
        return ()

    def run(self, plan, **model_inputs):
        return self.model(**model_inputs)
```

| Boundary               | Contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deployment and weights | `BuildContext` contains the worker's patched model, model/cache/backend configs, `DistContext` and device. Inspect actual local parameter layout, dtype, strides, quantization scales, layer count, sharding and ISA in `describe`. Return an empty tuple for unsupported deployments. Static discovery is allocation-free and collective-free. Preparation follows successful rank agreement and post-load processing, before initial KV sizing. V1 supplies processed local weights. A package needing raw checkpoint tensors must add a separately reviewed loader integration. |
| Shape and lengths      | `DecodePlan(name, batch_size, max_kv_len=None, graph_safe=False)` describes an **exact number of real requests per rank**, one query per row, and any set of positive KV lengths through the bound inclusive. `None` accepts the model's entire range. First matching declaration wins. A provider that requires a minimum or uniform length cannot express that property by checking a maximum. Keep such a plan out of coverage.                                                                                                                                                 |
| Inputs and output      | `run` accepts exactly the model's prepared kwargs. For GLM these include `input_ids [1,B]`, positions, `attn_metadata`, `past_key_values` and `state_ids`. Use the active context for the engine's device lengths, block table and bound named/state views. Return the same hidden-state tensor (GLM `[1,B,4096]`) or output dict, including any requested auxiliary fields. Logits, sampling and token publication belong to the caller.                                                                                                                                          |
| State                  | Respect each row's actual length, logical-to-physical page mapping, state input/output IDs, and invalid/reserved slots during warmup. Update the same buffers as standard execution exactly once. Borrow KV and auxiliary views for the call. The runner rebinds dynamic graph inputs. Persistent private request state would need an explicit migration/rollback contract.                                                                                                                                                                                                        |
| Ranks                  | Static plan lists and artifact identities must match before construction. Constructors prepare rank-local resources. After every worker reports success, `prepare_distributed(ctx)` may build symmetric buffers and exchange replicas in a fixed collective order using the engine's groups. Issue collectives in the same order for all participants. In DP mode one plan index travels with the existing CPU metadata exchange. A mismatch chooses the standard path for the group. A native program with device-side waits must supply a verified progress/failure protocol.    |
| Graphs and streams     | A graph-safe plan supports the runner's real warmup and synthetic startup capture with reserved dummy slots. Do not specialize on input tensor values in Python. Read replayable device buffers, finish preparation before capture, and join every side stream back to the caller's stream. The model owns graph input/output buffer construction. For graph-enabled serving, an eager-only plan leaves covered graphs with the standard forward.                                                                                                                                  |
| Lifetime and errors    | Keep derived weights and scratch on the instance. `close` runs after graph retirement. Refresh, sleep and shutdown must release those resources. Clean up partially constructed local resources on failure. Unsupported shapes fall back before model execution. Exceptions from `run` fail the step. Replaying a standard forward could apply writes twice. Current target dispatch excludes MTP, adapters, ReplaySSM and MemDecode.                                                                                                                                              |

## Build and run the H200 example

Use a working LMDeploy PyTorch/CUDA environment, eight H200s on one node, the real checkpoint and a CUDA 13 toolkit with driver headers, stubs and `ptxas`. The validation environment used PyTorch 2.12.1+cu130 and Triton 3.7.1. The example requires no R5–R10 binaries. Three whitespace-only lines were cleaned in the archived R11 PTX. CUDA 13.0 assembled both sources into the same cubin, SHA-256 `e2b6916d46ff7aa99240832ff091cd5affae67f1b79dd91e7446ce07410c54d0`. The host launcher adds explicit CUDA-module ownership. Preserve the original BF16 model projections for fallback. Packing costs 2.125 GiB per worker.

From the LMDeploy source root, point `MODEL` at the snapshot directory containing `config.json` and `model.safetensors.index.json`. Put the Python module and native directory at visible paths on **each worker**, including Ray workers.

```bash
export MODEL=/path/to/GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
export PYTHONPATH="$PWD:$PWD/examples/specialized_execution${PYTHONPATH:+:$PYTHONPATH}"
export GLM53_R11_PROJECTION_DIR=/tmp/glm53-r11-projection
bash examples/specialized_execution/native/build.sh "$GLM53_R11_PROJECTION_DIR"

export LMDEPLOY_DP_MASTER_ADDR=127.0.0.1
export LMDEPLOY_DP_MASTER_PORT=24640
lmdeploy serve api_server "$MODEL" --backend pytorch --distributed-executor-backend ray \
  --language-model-only --trust-remote-code --tp 1 --dp 8 --ep 8 \
  --specialized-executor glm53_r11:R11Projection --max-batch-size 8 \
  --cudagraph-capture-batch-sizes 8 --cache-block-seq-len 64 --server-port 24620
```

The DP launcher creates eight endpoints starting at the specified port. For the DeepGEMM configuration install the SM90 MegaMoE-capable DeepGEMM build used by #5186 on each worker and add `--moe-backend deepgemm_mega_moe`. Both the specialized and standard B8 graphs are prepared during engine warmup. B7 can use the standard padded B8 graph. A startup INFO message lists covered plans, capture logs contain their graph keys, and DEBUG logs identify actual specialized replays. [MegaMoE dependency](https://github.com/InternLM/lmdeploy/pull/5186)

## Verify an artifact

```bash
python -m pytest -q tests/pytorch/engine/test_specialized_execution.py

torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_glm53.py "$MODEL"
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_glm53.py "$MODEL" --eager --profile
torchrun --standalone --nproc-per-node=8 examples/specialized_execution/check_glm53.py "$MODEL" \
  --context 65536 --moe-backend deepgemm_mega_moe --prewarm --timed-steps 64
```

The worker test performs real prefill at three different lengths, saves the allocated pools and SSM cursor, then restores exactly the same state for the standard and candidate trajectories. It alternates request order, interleaves physical pages, and temporarily removes one row on rank zero. Each worker must select `[1,1,0,1,1]`, compare **all** logits at five steps byte-for-byte and compare **all five** final storage/cursor buffers. The capture test checks both graph identities. `--profile` also requires 34 actual `glm53_kda_output_r11` CUDA launches on rank zero. `--prewarm` uses normal ModelAgent startup warmup. The optional ABBA timer runs stock, specialized, specialized, stock from the same initial state, with one DP vote per step. CUDA events bracket model forward, LM head, argmax and feedback after that vote. Reduce the eight times by maximum **at each step** before calculating statistics.

## H200 validation

For the HTTP check add `--moe-backend deepgemm_mega_moe --model-name glm53-specialized --session-len 512 --max-prefill-token-num 1024 --num-gpu-blocks 256 --log-level DEBUG` to the server command. Set `DP_HOST` to the IP advertised in the server startup logs. This client sends the same single-request and balanced workloads.

```python
import asyncio
import os
import aiohttp

async def smoke():
    async with aiohttp.ClientSession() as client:
        async def generate(rank, request_id, length):
            body = dict(model="glm53-specialized", prompt=f"Test request {request_id:03d}. Count from one to thirty-two in English.",
                        temperature=0, max_tokens=length, ignore_eos=True, stream=False)
            url = f"http://{os.environ['DP_HOST']}:{24620 + rank}/v1/completions"
            async with client.post(url, json=body) as response:
                response.raise_for_status()
                assert (await response.json())["usage"]["completion_tokens"] == length

        await generate(0, 1, 4)
        for _ in range(2):
            await asyncio.gather(*(generate(rank, rank * 8 + row, 32) for rank in range(8) for row in range(8)))

asyncio.run(smoke())
```

| Run                                    | Input / setting                                                                                                        | Observed result                                                                                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eager and Full Graph, default MoE      | 8 workers, prefill lengths 63–65, physical pages crossing the 64-token boundary                                        | Both passed on all ranks. Eager profiler counted 34 native projections.                                                                                                     |
| Full Graph with #5186 DeepGEMM MegaMoE | Prefill lengths 65,536–65,538, cold capture and separate startup-prewarmed runs                                        | Full logits and five final buffers matched on every rank. Rank-zero prewarmed replay had 34 native projections.                                                             |
| HTTP through the normal scheduler      | Eight Ray-backed endpoints. One request, then two waves of 8 requests per endpoint                                     | 129 successful text completions, 4,100 output tokens total. All eight workers logged specialized replay during live serving.                                                |
| Backend regression                     | Host-side Ascend dispatch/graph tests plus PyTorch configuration, compact allocation, GLM, model agent and graph tests | 683 tests passed. One on-device Ascend import test was skipped because it requires the vendor runtime. An A3 hardware run and long-duration serving SLOs remain unmeasured. |

With the same DeepGEMM MoE in both runners, the unprofiled 64-step ABBA runs gave rank-max means of **18.095 / 17.961 / 17.955 / 18.051 ms**. Combining the two arms per route gives **18.073 ms stock, 17.958 ms specialized** (0.115 ms, 0.64% lower). Pooled p50 was 18.211 / 18.109 ms and p95 was 18.519 / 18.402 ms. A separate 32-step diagnostic immediately after a profiler pass had an 88.178 ms stock observation. The accompanying measurements retain that observation. These are fixed-workload worker timings. The HTTP run checked request completion and actual routing. Historical whole-R11 numbers use a different model body and device-controlled loop. [Measurements](validation.json)
