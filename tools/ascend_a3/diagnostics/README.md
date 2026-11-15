# Ascend A3 investigation tools

These tools support the GDN state-write and CPU 18 investigation. They do not
change LMDeploy's sampling or synchronization, IRQ affinity, driver settings,
or kernel sysctls. CPU affinity avoidance is a temporary mitigation for the
observed host, not a permanent fix.

## Capture the actual tasklet callback on the host

Run on the **A3 host**, with permission to create tracefs instances/kprobes and
read kernel symbol addresses. Container UID 0 alone is insufficient. Python
3.9+, live `/sys/kernel/btf/vmlinux`, tracefs, and `CONFIG_KPROBE_EVENTS` are
required. The optional `--perf-stacks` also requires the host `perf` tool and
permission for kernel CPU sampling.

If tracefs is not already mounted, the host administrator can mount it:

```bash
sudo mount -t tracefs tracefs /sys/kernel/tracing
```

From the repository root, use a fresh output directory for every run:

```bash
TASKLET_OUT=/var/tmp/a3-tasklet-$(date -u +%Y%m%dT%H%M%SZ)
sudo python3 tools/ascend_a3/diagnostics/capture_tasklet.py \
  --cpu 18 --seconds 40 --discover-seconds 10 \
  --perf-stacks --output "$TASKLET_OUT"
sudo python3 tools/ascend_a3/diagnostics/analyze_tasklet.py "$TASKLET_OUT"
```

Omit `--perf-stacks` if the tool is unavailable; timed callback probes still
work, but instruction/call-stack attribution will remain incomplete. A
non-invasive preflight is available with `--check-only` and a separate output
directory. This checks actual permissions and symbol visibility instead of
assuming that `sudo` or a container mount is sufficient.

The collector uses two phases:

1. Read the live BTF layout of `tasklet_struct`; capture `__tasklet_schedule`
   and `__tasklet_hi_schedule`, including the callback pointer, tasklet,
   argument/data, and callback ABI flag. Do not hard-code a driver callback.
2. Resolve exact callback addresses using the host's `kallsyms`. Attach entry
   and return probes to those functions, alongside IRQ and softirq events.
   The analyzer matches callback arguments back to scheduled tasklets and the
   IRQ active when they were scheduled, and computes callback duration within
   each slow softirq interval. It separately accounts for recorded nested IRQs.

The instance and probe names are unique. All created probes and the private
instance are removed on normal exit, SIGINT, or SIGTERM; cleanup failures are
recorded. Recording is restricted to the requested CPU. Kprobe insertion is
global, so calls on other CPUs still incur probe overhead. Duration, buffer
size, trace bytes, and the number of callbacks are bounded. Existing tracing
facilities are not cleared or disabled. A SIGKILL or host crash cannot run
cleanup; `metadata.json` or `preflight.json` identifies this run's unique group.

Inspect these outputs before drawing a conclusion:

- `callbacks.json`: resolved callback function/module and attachment status.
- `analysis.json`: complete event pairs, slow softirqs, associated tasklets/IRQs,
  unmatched boundary events, missed probes, and trace overruns.
- `trace.txt`, `cpu-softirq.jsonl`, `interrupts-*.txt`, `modules-*.txt`: raw evidence.
- `perf.data` / `perf-script.txt`: optional kernel stacks during the stalls.
- `metadata.json`: command, kernel, BTF hash, clock, losses, failures, and cleanup.

Discovery only covers callbacks observed during its first phase. Newly appearing
callbacks, unresolved symbols, probe attachment failures, buffer overruns, missed
kretprobes, or truncated event boundaries must remain explicit limitations.
An IRQ name or scheduling count alone does not identify the slow callback.
Kernel sampling and nested IRQs must also be checked before treating inclusive
callback time as time spent in the callback's own instructions.

On the investigated container, only syntax, actual BTF decoding, synthetic
event-pairing/IRQ association, and rejection of insufficient privileges have
been validated. **Host callback collection has not been executed there.** The
kernel callback and its permanent fix remain unresolved until that evidence is
collected. CPU 18 is a host-specific observation; do not assume it is problematic
on another host or architecture.

## Standalone compiler reproducer

`reproduce_state_writeback.py` uses only Torch, torch_npu, and Triton Ascend.
It does not load a model, Ray, LMDeploy, or dlinfer. Use the investigation's
fixed native environment and run from a checkout where dlinfer is not visible.

```bash
python tools/ascend_a3/diagnostics/reproduce_state_writeback.py \
  --stages 3 --iterations 100 --output /tmp/state-multibuffer-new.json
python tools/ascend_a3/diagnostics/reproduce_state_writeback.py \
  --stages 1 --iterations 100 --output /tmp/state-singlebuffer-new.json
```

The recurrence is `h = 0.5*h + x*q`; every intermediate state is saved. On the
fixed A3 compiler, automatic multi-buffering corrupted intermediate history in
100/100 graph replays. Single buffering produced zero state errors. This tests
stored history separately from the current output.

The compiler defect is a mismatch in rotation scope: GraphSyncSolver records the
lexical loop of a pointer, while EnableMultiBuffer's yielded-value-aware helper
may choose an outer loop. A loop-carried state buffer can therefore retain one
physical address across token iterations while its reuse events alternate IDs.
The subsequent vector update can overwrite data still read by an MTE3 store.
Two independent IR repairs (matching the event ID to actual reuse, or matching
the state address to the existing event rotation) passed fixed-input replay.
Those IR edits are diagnostic artifacts, not installed compiler replacements.

The LMDeploy workaround disables auto multi-buffering for circular GDN states.
The compiler source candidate needs a separate build and full compiler regression
before use. Microbenchmarks and a frozen target forward are distinct from online
throughput; do not generalize one shape's timing to complete model serving.

Full version pins, commands and evidence are maintained in the fork's
[investigation report](https://github.com/silencelamb/lmdeploy/blob/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation/mtp_ray_investigation_20261116.md).
