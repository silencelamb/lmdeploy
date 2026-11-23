"""Plot the measured 1,800-second QA ladder and its reported historical reference points."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(source, destination):
    data = json.loads(source.read_text())
    points = data["points"]
    levels = [r["concurrency"] for r in points]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    style = {
        "current": ("#087d5b", "o", "MTP3 + Mooncake (current)"),
        "lmdeploy": ("#777777", "s", "LMDeploy (QA)"),
        "vllm_0_33_0": ("#3565ad", "^", "vLLM 0.33.0 (QA)"),
    }
    curves = [
        ("TTFT p50 (s)", lambda r: r["ttft_ms"]["p50"] / 1000, lambda r: r["ttft_p50_ms"] / 1000, 3),
        ("TPOT p50 (ms)", lambda r: r["tpot_ms"]["p50"], None, 25),
        ("Output throughput (tokens/s)", lambda r: r["decode_tok_s"], lambda r: r["decode_tok_s"], None),
        ("Per-request SLO attainment (%)", lambda r: 100 * r["attainment"], lambda r: r["attainment_percent"], None),
    ]
    for ax, (label, current, historical, limit) in zip(axes.flat, curves):
        for key in ("lmdeploy", "vllm_0_33_0"):
            pairs = [(c, data["comparison"][str(c)][key]["old"]) for c in levels if key in data["comparison"][str(c)]]
            if historical is not None and pairs:
                color, mark, _ = style[key]
                ax.plot(
                    [c for c, _ in pairs],
                    [historical(r) for _, r in pairs],
                    color=color,
                    marker=mark,
                    linestyle="--",
                    linewidth=1.4,
                    markersize=5,
                )
        color, mark, _ = style["current"]
        ax.plot(levels, [current(r) for r in points], color=color, marker=mark, linewidth=2, markersize=6, zorder=3)
        failed = [(r["concurrency"], current(r)) for r in points if not r["both_medians_pass"]]
        if failed:
            ax.scatter(*zip(*failed), s=85, facecolors="none", edgecolors="#ac4327", linewidths=1.4, zorder=4)
        if limit is not None:
            ax.axhline(limit, linewidth=1, ls=":", color="#ac4327")
        ax.set(ylabel=label, xlabel="Concurrent sessions", xticks=levels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if label.endswith("(%)"):
            ax.set_ylim(0, 100)
    handles = [
        Line2D([0], [0], color=c, marker=m, ls="-" if k == "current" else "--", label=l)
        for k, (c, m, l) in style.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    draw(args.source, args.output)
