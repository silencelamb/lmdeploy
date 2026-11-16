# A3 AscendNPU-IR 2.0.0 validation profile

This research branch validates a dependency update. The LMDeploy computational
source remains native commit `5f1b1b3c12b1c198813f65275f778c7ad292fb02`, including
`num_stages=3` for circular GDN state. The migration PR and original environment
are unchanged. Do not merge the old single-buffer workaround into this profile
and then claim to have validated multibuffering.

The complete report and recorded commands are maintained on the
[fork documentation branch](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/mtp_ray_investigation_20261116.md).
The new evidence directory is `evidence/npuir2-upgrade-20261116` beside that report.

## Pinned components

| Component | Original | Validated upgrade |
| --- | --- | --- |
| Python | 3.12.13 / aarch64 | unchanged |
| Torch / TorchNPU | 2.10.0+cpu / 2.10.0.post4 | unchanged |
| CANN runtime / driver | 9.1.0 / npu-smi 25.5.6 | unchanged |
| Triton-Ascend distribution | 3.2.2 | 3.2.2.post2 |
| Actual bundled NPUIR | 1.2.0 / 042c923d02a2 | 2.0.0 / efb7141581c0 |

Official wheel:

```text
https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.2.post2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
SHA256 ca16e1f6df5850663e430789e1d68462d902bb4701a6f0441c15e175bb132005
```

The wheel's `bishengir-compile`, `bishengir-opt`, and `hivmc` report the exact
AscendNPU-IR v2.0.0 tag, `efb7141581c009bbd1d676b18941ed15ede62a68`, which contains
upstream fix `05f2adcf3db366377f7c5e6f5bb9bd3c61c36b3e`. The original compiler
comes from the `feature/0725` / `a3_fix_kda` line and does not contain that fix.

Triton-Ascend translates a Triton kernel and invokes the bundled NPUIR compiler;
NPUIR supplies lower-level optimization and NPU code generation. TorchNPU/CANN
provide the execution environment. These are distinct versioned components.
`triton.__version__` alone does not identify the Ascend compiler. In this setup,
the backend prefers `triton/backends/ascend/bishengir/bin/bishengir-compile` over
PATH, so installing a separate compiler on PATH is insufficient.

## Reproduction on the investigation machine

The original environment is preserved. These two isolated environments exist:

- `/root/workspace/a3-mtp-ray-investigation-20261116/envs/upgrade5-ir2`: original
  frontend with only a private compiler-bundle replacement.
- `/root/workspace/a3-mtp-ray-investigation-20261116/envs/upgrade5-full322p2`:
  complete official wheel, with all other dependencies held fixed.

Unchanged dependencies are reused through symlinks. **Do not run pip install or
uninstall inside these linked validation environments.** The archived
`upgrade5_make_env.py` and `upgrade5_extract_wheel.py` create fresh directories,
avoid mutating the original package tree, and record the actual binary hashes.
They deliberately retain the existing NumPy and other runtime dependencies;
this is not a claim that fresh unrestricted pip dependency resolution was tested.

Use a new label and output filename for every rerun:

```bash
A3_UPGRADE_ROOT=/root/workspace/a3-mtp-ray-investigation-20261116
python "$A3_UPGRADE_ROOT/scripts/run_upgrade5.py" \
  --venv "$A3_UPGRADE_ROOT/envs/upgrade5-full322p2" \
  --compiler-cache upgrade5-full322p2 \
  --label upgrade5-recheck-new --devices 0 -- \
  "$A3_UPGRADE_ROOT/lmdeploy-research/tools/ascend_a3/diagnostics/reproduce_state_writeback.py" \
  --stages 3 --iterations 100 --output "$A3_UPGRADE_ROOT/results/upgrade5-recheck-new.json"
```

The launcher sets the selected native checkout as cwd, enables the startup
guard against discovering/importing dlinfer, records the source and environment,
and excludes CPU18 for these experiments. CPU affinity is a temporary control
for this host's separately observed stalls, not a host/Ray fix.

Correctness checks passed on the A3: standalone stages3 state history 0/100
errors (original compiler 100/100), real frozen GDN state/output 200/200 bitwise
matches, six Q1/2/4 eager/graph hardware cases on both upgrade profiles, and two
fresh Sage TP2/MTP3 graph/prefix runs per profile. Each run's 21 responses and
top5 scores exactly match the previous correct single-buffer result.

The GDN benchmark compares **two correct implementations** under NPUIR2.0.0.
Multibuffering is shape-dependent: B4/Q4 is about 2.51% faster than single-buffer,
while B16/Q4 is about 4.83% slower. See the report for frozen target-backbone
timings and exact measurement scope; do not infer HTTP throughput from them.

The official preferred NPUIR2.0.0 pairing is CANN9.2.0. These experiments validate
the narrower component update on the existing CANN9.1.0 runtime. Full CANN9.2.0,
driver replacement, other models, long contexts, high concurrency, and end-to-end
HTTP throughput are outside the validated scope. The downloaded CANN9.2.0
toolkit was kept as an unused fallback and was not installed.
