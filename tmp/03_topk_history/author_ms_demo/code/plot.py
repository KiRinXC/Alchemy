"""绘制作者排序下的 Best 与 Epoch 100 准确率，支持首阶段及完整扫描。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_results(folder: Path) -> Path:
    """Best 为主曲线，Epoch 100 为辅助曲线；首阶段只展示已测点。"""
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("evaluation") != "official_test_used_for_best_epoch_and_topk_selection":
        raise ValueError("绘图需要按 Best Test Accuracy 选模的结果。")
    with (folder / "accuracy.tsv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    groups = {strategy: sorted((row for row in rows if row["strategy"] == strategy), key=lambda row: int(row["k"]))
              for strategy in ("freeze", "joint")}
    ks = [int(row["k"]) for row in groups["freeze"]]
    if not ks or len(ks) != len(set(ks)) or ks != [int(row["k"]) for row in groups["joint"]]:
        raise ValueError("绘图需要两种策略无重复、对应的 Top-k 结果。")
    if summary["scan_complete"] and ks != list(range(1, summary["candidate_count"] + 1)):
        raise ValueError("完整扫描缺少 Top-k 结果。")
    if any(left["protected_ratio"] != right["protected_ratio"]
           for left, right in zip(groups["freeze"], groups["joint"])):
        raise ValueError("两种策略的保护比例不一致。")
    blackbox, victim = summary["all_shield_accuracy"], summary["victim_accuracy"]
    upper = min(100, 5 * math.ceil(max(victim, summary["threshold_accuracy"],
                                     *(float(row["best_accuracy"]) for group in groups.values() for row in group)) / 5) + 5)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
        for axis, strategy, color in zip(axes, ("joint", "freeze"), ("#C46B28", "#2878B5")):
            accuracies = [float(row["best_accuracy"]) for row in groups[strategy]]
            axis.plot(ks, accuracies, marker="o", markersize=4, color=color,
                      linestyle="-" if summary["scan_complete"] else "none", label="Best test accuracy")
            axis.plot(ks, [float(row["final_accuracy"]) for row in groups[strategy]], marker="x", color=color,
                      alpha=0.5, linestyle=":" if summary["scan_complete"] else "none", label="Epoch 100")
            axis.axhline(blackbox, color="#B33B45", linestyle="--", label=f"All-Shield Best: {blackbox:.2f}%")
            axis.axhline(victim, color="#444444", linestyle="-.", label=f"Victim: {victim:.2f}%")
            axis.axhline(summary["threshold_accuracy"], color="#888888", linestyle=":",
                         label=f"All-Shield + {summary['epsilon_pp']:g} pp")
            critical = summary["critical_k"][strategy]
            if critical is not None:
                index = ks.index(critical)
                axis.scatter([critical], [accuracies[index]], s=110, facecolors="none", edgecolors="#239B56", linewidths=2, zorder=4)
                axis.axvline(critical, color="#239B56", linestyle=":", alpha=0.65)
                ratio = float(groups[strategy][index]["protected_ratio"])
                caption = f"Minimum k = {critical}; protected = {100 * ratio:.2f}%"
            elif summary["scan_complete"]:
                caption = "Black-box threshold not reached"
            else:
                caption = "Probe only; global minimum pending"
            role = "primary" if strategy == "joint" else "auxiliary"
            axis.set_title(f"{strategy.title()} ({role}) | {caption}", loc="left", fontsize=11)
            axis.set_ylabel("Test accuracy (%)")
            axis.set_ylim(0, upper)
            axis.yaxis.set_major_locator(MultipleLocator(10))
            axis.grid(axis="y", alpha=0.2)
            axis.legend(loc="upper right", ncol=2, fontsize=8)
        axes[-1].set_xlim(0.6, summary["candidate_count"] + 0.4)
        axes[-1].set_xticks(ks, [f"{k}\n{100 * float(row['protected_ratio']):.2f}%"
                                for k, row in zip(ks, groups["freeze"])], fontsize=8)
        axes[-1].set_xlabel("Author Top-k + fixed head bias / protected parameters (%)")
        fig.suptitle("Official test set used for best-epoch and Top-k selection", fontsize=10)
        destination = folder / "topk_accuracy.png"
        temporary = destination.with_suffix(".png.tmp")
        try:
            fig.savefig(temporary, format="png", dpi=180)
            temporary.replace(destination)
        finally:
            plt.close(fig)
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="包含 accuracy.tsv 和 summary.json 的单次结果目录")
    args = parser.parse_args()
    print(plot_results(args.results))


if __name__ == "__main__":
    main()
