# 性能数字复核脚本

只读取保存的 JSON 样本，输出统计复核，不加载 Torch 或运行模型。

将下列代码保存为 `review_performance.py`，在仓库根目录执行：

```bash
python3 review_performance.py \
  --docs-root docs/zh_cn/advance/ascend_a3_validation \
  --output-dir /tmp/pr5134-performance-review
```

```python
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--docs-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
docs = args.docs_root
outdir = args.output_dir
outdir.mkdir(parents=True, exist_ok=True)
inputs = {}
audit = {"date": "2026-11-17", "new_device_runs": False, "checks": [], "h200_adverse_median_over_5pct": [], "a3_historical_http": [], "a3_norm_paired": []}
def read(rel):
    p = docs / rel
    data = p.read_bytes()
    inputs[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return json.loads(data)
def eq(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)
total_metrics = 0
triggered = []
for rel in ["evidence/cuda-h200-20261116/performance-summary.json", "evidence/cuda-h200-20261116/decision-performance-summary.json"]:
    groups = read(rel)
    for group in groups:
        for metric, data in group["metrics"].items():
            total_metrics += 1
            changes = [(p / b - 1) * 100 for b, p in zip(data["base"], data["pr"])]
            assert len(changes) == 5
            for a, b in zip(changes, data["paired_pct"]):
                eq(a, b)
            med = statistics.median(changes)
            eq(med, data["paired_median_pct"])
            if "base_median" in data:
                eq(statistics.median(data["base"]), data["base_median"])
                eq(statistics.median(data["pr"]), data["pr_median"])
            ci = data.get("paired_bootstrap_ci95_pct", data.get("ci95"))
            sign = -1 if metric in ("throughput", "rps", "prefill_token_tps") else 1
            adverse = [sign * c for c in changes]
            ci_adverse = sorted(sign * x for x in ci)
            needs = sign * med > 5 and sum(x > 5 for x in adverse) >= 4 and ci_adverse[0] > 0
            assert needs == data["requires_investigation"]
            identity = {k: group.get(k) for k in ("model", "mtp", "scope", "concurrency", "timing_profile")}
            identity["model"] = identity["model"] or "decision"
            item = {**identity, "metric": metric, "comparable": group.get("comparable", group.get("workload_equal")),
                    "base_median": statistics.median(data["base"]), "pr_median": statistics.median(data["pr"]),
                    "ratio_of_medians_pct": (statistics.median(data["pr"]) / statistics.median(data["base"]) - 1) * 100,
                    "paired_pct": changes, "paired_median_pct": med, "recorded_ci95_pct": ci,
                    "adverse_pairs_over_5pct": sum(x > 5 for x in adverse),
                    "requires_investigation": needs}
            if sign * med > 5:
                audit["h200_adverse_median_over_5pct"].append(item)
            if needs:
                triggered.append(item)
    audit["checks"].append({"file": rel, "groups": len(groups), "all_paired_values_medians_and_trigger_flags_recomputed": True, "ci95_read_from_recorded_summary": True})

audit["h200_metric_groups_checked"] = total_metrics
audit["h200_triggered_groups"] = triggered
for model, names in {
    "sage": ["compare-http-fresh.json", "compare-http-recheck.json"],
    "agents": ["compare-http-agents.json", "compare-http-agents-recheck.json"],
}.items():
    for order, name in zip(["baseline_then_native", "native_then_baseline"], names):
        d = read("evidence/20261112/results/" + name)
        for concurrency, s in d["scenarios"].items():
            changes = {}
            for key, stored in s["change_percent"].items():
                value = (s["native"][key] / s["baseline"][key] - 1) * 100
                eq(value, stored)
                changes[key] = value
            for side in ("baseline", "native"):
                eq(statistics.median(s[side]["throughput_repetitions"]), s[side]["median_throughput"])
            audit["a3_historical_http"].append({
                "model": model, "order": order, "concurrency": int(concurrency),
                "repetitions": s["baseline"]["repetitions"], "changes_pct": changes,
                "source": "evidence/20261112/results/" + name,
            })
d = read("evidence/norm-perf-20261116/results/agents-paired-gc-observe.summary.json")
for m in d["metrics"]:
    vals = [(b / a - 1) * 100 for a, b in zip(m["samples"]["old"], m["samples"]["new"])]
    eq(statistics.median(vals), m["paired_median_change_pct"])
    for a, b in zip(vals, m["paired_change_pct"]):
        eq(a, b)
    for side in ("old", "new"):
        eq(statistics.median(m["samples"][side]), m["medians"][side])
    eq((m["medians"]["new"] / m["medians"]["old"] - 1) * 100, m["median_change_pct"])
    audit["a3_norm_paired"].append({k: m[k] for k in ("batch", "metric", "median_change_pct", "paired_median_change_pct", "paired_median_bootstrap_95_pct")})
audit["inputs"] = inputs
audit["limitations"] = [
    "Recomputed statistics from preserved samples; no new GPU/NPU timing.",
    "Recorded bootstrap intervals retained; not interpreted as equivalence bounds.",
    "Historical A3 dlinfer-to-native HTTP slowdown signals have not been closed by native-only norm or CPU-affinity controls.",
    "All original raw results and fixed source revisions remain unchanged.",
]
payload = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
(outdir / "audit.json").write_text(payload)
print(json.dumps({"metrics_checked": total_metrics, "triggered": len(triggered), "adverse_median_groups": len(audit["h200_adverse_median_over_5pct"]), "a3_http_scenarios": len(audit["a3_historical_http"]), "a3_norm_metrics": len(audit["a3_norm_paired"]), "audit_sha256": hashlib.sha256(payload.encode()).hexdigest()}, ensure_ascii=False))
for item in audit["h200_adverse_median_over_5pct"]:
    print(json.dumps(item, ensure_ascii=False))
```
