"""从完整汇总绘制最佳准确率、相对黑盒差距及其相对 B0 的变化。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
import tomli

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ("#1e293b", "#b64435", "#266cad", "#936c13", "#157a63", "#834ca6")
MARKERS = ("o", "s", "^", "D", "P", "v")


def default_results(settings: dict) -> Path:
    return (ROOT / "results/lab/01_basecheck" / settings["model"] / settings["dataset"]
            / f"a_seed_{settings['a_seed']}" / f"b_seed_{settings['b_seed']}")


def plot_results(folder: Path) -> list[Path]:
    """使用本次扫描的观测点和各配置自己的 All-Shield 基线。"""
    with (HERE / "config.toml").open("rb") as stream:
        settings = tomli.load(stream)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["evaluation"] != "official_test_used_for_best_epoch_and_topk_selection":
        raise ValueError("绘图需要按官方测试集选择 Best 轮次的结果。")
    ks = settings["k"]
    variants = [f"B{index}" for index in range(6)]
    scenarios = ("", *(str(k) for k in ks))
    with (folder / "accuracy.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    indexed = {(row["variant"], row["k"]): row for row in rows}
    if len(rows) != len(indexed) or indexed.keys() != {
        (variant, k) for variant in variants for k in scenarios
    } or not all(summary["variants"][variant]["scan_complete"] for variant in variants):
        raise ValueError("绘图需要完整且无重复的六配置、九场景结果。")

    plots = (
        ("best_accuracy", "Best test accuracy (%)", "Best attack accuracy", variants, None),
        ("gap_to_all_shield", "Best accuracy minus All-Shield (pp)", "Gap to matched All-Shield",
         variants, (summary["epsilon_pp"], f"Threshold (+{summary['epsilon_pp']:g} pp)")),
        ("delta_gap", "Gap change vs B0 (pp)", "Change in gap relative to B0",
         variants[1:], (0.0, "B0 reference")),
    )
    destinations = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        for metric, ylabel, title, lines, reference in plots:
            fig, axis = plt.subplots(figsize=(9.4, 5.8), layout="constrained")
            destination = folder / f"{metric}.png"
            temporary = destination.with_suffix(".png.tmp")
            try:
                for variant in lines:
                    index = variants.index(variant)
                    values = [float(indexed[variant, str(k)][metric]) for k in ks]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"绘图指标存在非有限值：{variant} {metric}。")
                    label = f"{variant} {settings['variants'][variant]['name']}"
                    if metric == "best_accuracy":
                        blackbox = float(indexed[variant, ""][metric])
                        axis.axhline(blackbox, color=COLORS[index], linestyle="--", linewidth=0.9, alpha=0.6)
                        label += f" ({blackbox:.2f}% All-Shield)"
                    axis.plot(ks, values, color=COLORS[index], marker=MARKERS[index],
                              linewidth=1.6, markersize=5, label=label)
                if reference:
                    value, label = reference
                    axis.axhline(value, color="#6b7280", linestyle="--", linewidth=1, label=label)
                axis.set_title(title, loc="left", fontweight="bold")
                axis.set_xlabel("Author Top-k (Joint, fixed head bias)")
                axis.set_ylabel(ylabel)
                axis.set_xticks(ks)
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
                axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2 if metric == "best_accuracy" else 3,
                            frameon=False, fontsize=9)
                fig.suptitle("Official test set used for best-epoch and Top-k selection", fontsize=9, color="#475569")
                fig.savefig(temporary, format="png", dpi=180)
                temporary.replace(destination)
            finally:
                plt.close(fig)
                temporary.unlink(missing_ok=True)
            destinations.append(destination)
    return destinations


def main() -> None:
    with (HERE / "config.toml").open("rb") as stream:
        settings = tomli.load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=default_results(settings),
                        help="包含完整 accuracy.tsv 和 summary.json 的结果目录")
    args = parser.parse_args()
    for path in plot_results(args.results):
        print(path)


if __name__ == "__main__":
    main()
