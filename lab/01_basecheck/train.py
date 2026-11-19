"""沿用 Exp03 的数据与训练逻辑，对照分类头初始化、学习率与权重衰减。"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import importlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import tomli
import torch

from config import load_config
from models.build import build_model
from randomness import preserve_rng_state, seed_everything

exp03 = importlib.import_module("exp.03_topk.train")
base = exp03.base
LAB = Path("lab/01_basecheck")
EXP03 = Path("exp/03_topk")
EVALUATION = "official_test_used_for_best_epoch_and_topk_selection"
HEAD_NAMES = {"last_linear.weight", "last_linear.bias"}
RECORD_FIELDS = exp03.ACCURACY_FIELDS[:-1]
ACCURACY_FIELDS = (
    "variant", *exp03.ACCURACY_FIELDS, "final_gap_to_all_shield", "delta_gap", "final_delta_gap", "source",
)
HISTORY_FIELDS = ("variant", *exp03.HISTORY_FIELDS)
METRICS = ("best_accuracy", "final_accuracy", "gap_to_all_shield", "final_gap_to_all_shield")


def read_settings() -> tuple[dict, dict, dict]:
    with (HERE / "config.toml").open("rb") as stream:
        settings = tomli.load(stream)
    variants = settings.pop("variants")
    if (settings["model"], settings["dataset"], settings["a_seed"], settings["b_seed"], settings["query_seed"]) != (
        "resnet18", "c100", 42, 42, 42,
    ):
        raise ValueError("Lab01 固定使用 ResNet18/CIFAR-100 和 A、B、查询种子 42。")
    if settings["k"] != list(range(10, 18)) or settings["epsilon_pp"] != 1.0:
        raise ValueError("Lab01 固定扫描 Top-10～Top-17，黑盒容差为 1 pp。")
    matrix = {
        "B0": ("B", 0.1, 0.0005), "B1": ("A", 0.1, 0.0005),
        "B2": ("B", 0.01, 0.0005), "B3": ("B", 0.1, 0.0),
        "B4": ("B", 0.01, 0.0), "B5": ("A", 0.01, 0.0),
    }
    if set(variants) != set(matrix) or any(
        tuple(variants[key][field] for field in ("head", "lr", "weight_decay")) != values
        for key, values in matrix.items()
    ):
        raise ValueError("配置矩阵与 Lab01 协议不符。")
    formal = exp03.read_settings(ROOT / EXP03 / "config.toml")
    training = exp03.read_training(ROOT / formal["training_config"])
    if (training["epochs"], training["batch_size"], training["lr"], training["weight_decay"],
        training["momentum"], training["step_size"], training["gamma"]) != (100, 64, 0.1, 0.0005, 0.5, 60, 0.1):
        raise ValueError("Exp03 的训练协议与 Lab01 预期不符。")
    settings.update({key: formal[key] for key in ("num_workers", "cpu_threads", "device", "victim")})
    return settings, variants, training


def run_path(settings: dict) -> Path:
    return (Path(settings["model"]) / settings["dataset"] / f"a_seed_{settings['a_seed']}"
            / f"b_seed_{settings['b_seed']}")


def build_publics(settings, a_config, b_config) -> dict:
    """保留 Exp03 的 B 初始化，另外只用 A 微调前的两个头张量构造 A-Head。"""
    public_b = exp03.build_public(settings, b_config)
    with preserve_rng_state():
        a_init = build_model(settings["model"], settings["dataset"], a_config)
        public_a = copy.deepcopy(public_b)
        public_a.last_linear.load_state_dict(a_init.last_linear.state_dict(), strict=True)
    return {"A": public_a, "B": public_b}


def load_inputs(settings, a_config, query_config) -> tuple:
    victim, victim_path = exp03.load_victim(settings, a_config)
    model, dataset = settings["model"], settings["dataset"]
    rank_path = ROOT / "results/exp/02_rank" / model / dataset / f"seed_{a_config.seed}" / "rank.tsv"
    rank = exp03.load_rank(rank_path, victim)
    ratio = f"ratio_{query_config.ratio_name}"
    indices_path = ROOT / "dataset/splits" / dataset / f"seed_{query_config.seed}" / ratio / "b_indices.txt"
    indices = exp03.read_indices(indices_path, query_config.b_ratio)
    if len(indices) != 500:
        raise ValueError("Lab01 固定复用 500 张查询图片及其软标签。")
    query_path = (ROOT / "dataset/queries" / run_path(settings).parent
                  / f"query_seed_{query_config.seed}" / ratio / "probabilities.pth")
    cached = exp03.load_queries(query_path, indices, indices_path, victim, victim_path, settings)
    inputs = {key: exp03.identity(path) for key, path in {
        "victim": victim_path, "rank": rank_path, "query_indices": indices_path, "query_cache": query_path,
    }.items()}
    return victim, rank, cached, inputs


def lab_protection(public, rank: list[str], ks: list[int]) -> dict:
    rows = {row["k"]: row for row in exp03.protection_rows(public, rank)}
    selected = {k: dict(rows[k], strategy="all_shield" if k is None else "joint") for k in (None, *ks)}
    if any(not HEAD_NAMES <= set(row["protected_tensors"].split(";")) for row in selected.values()):
        raise ValueError("扫描范围内必须保护完整分类头。")
    return selected


def same_state(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(value, right[key]) for key, value in left.items())


def check_initialization(publics: dict, victim, protection: dict) -> None:
    before_b = publics["B"].state_dict()
    before_a = publics["A"].state_dict()
    if {name for name in before_b if not torch.equal(before_b[name], before_a[name])} != HEAD_NAMES:
        raise ValueError("两种公共初始化应仅在分类头 weight 和 bias 上有差异。")
    victim_state = victim.state_dict()
    for public in publics.values():
        public_state = public.state_dict()
        for k, row in protection.items():
            protected = set(row["protected_tensors"].split(";"))
            student = exp03.make_student(public, victim, protected, row["strategy"])
            expected = public_state if k is None else {
                name: public_state[name] if name in protected else value
                for name, value in victim_state.items()
            }
            if not same_state(student.state_dict(), expected) or any(
                not parameter.requires_grad for parameter in student.parameters()
            ):
                raise ValueError(f"初始参数、BN buffers 或可训练状态不符：{row['strategy']} k={k}。")


def group_key(row: dict) -> tuple:
    return row["strategy"], row["k"]


def read_rows(path: Path, fields: tuple, integer: tuple, delimiter=",") -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: (int(row[key]) if row[key] else None) if key in ("k", *integer)
             else row[key] if key == "strategy" else float(row[key]) for key in fields}
            for row in csv.DictReader(stream, delimiter=delimiter)
        ]


def check_group(record: dict, history: list[dict], training: dict, protection: dict) -> None:
    group = group_key(record)
    epochs = training["epochs"]
    if len(history) != epochs or any(
        group_key(row) != group or row["epoch"] != epoch or any(
            not math.isfinite(row[key]) for key in ("lr", "train_loss", "test_accuracy", "seconds")
        ) or not (0 <= row["test_accuracy"] <= 100) or not math.isclose(
            row["lr"], training["lr"] * training["gamma"] ** ((epoch - 1) // training["step_size"])
        )
        for epoch, row in enumerate(history, start=1)
    ):
        raise ValueError(f"逐轮历史、分组或学习率不符：{group}。")
    best = max(history, key=lambda row: (row["test_accuracy"], row["epoch"]))
    if (record["best_accuracy"], record["best_epoch"], record["final_accuracy"]) != (
        best["test_accuracy"], best["epoch"], history[-1]["test_accuracy"],
    ) or not 0 <= record["initial_accuracy"] <= 100 or any(
        record[key] != protection[key] for key in ("protected_parameters", "protected_ratio")
    ):
        raise ValueError(f"准确率摘要或保护参数量不符：{group}。")


def load_baseline(settings, b_config, training, public, inputs, protection) -> tuple:
    """直接读取已完成 Exp03 的九组结果，核对其训练记录、公共初始化和输入。"""
    folder = ROOT / "results" / EXP03 / run_path(settings)
    weights = ROOT / "weights" / EXP03 / run_path(settings)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    baseline_inputs = {**inputs, "public_init": exp03.identity(weights / "public_init.pth")}
    progress = torch.load(weights / "progress.pth", map_location="cpu", weights_only=True)
    source_config = progress["configuration"]
    expected = {
        "inputs": baseline_inputs, "training": training, "a_seed": settings["a_seed"],
        "b_seed": b_config.seed, "query_seed": settings["query_seed"], "b_ratio": b_config.ratio_name,
        "deterministic": b_config.deterministic, "head_seed": exp03.head_seed(settings, b_config),
        "rank_field": exp03.RANK_FIELD, "fixed_protected": sorted(exp03.FIXED_PROTECTED),
        "victim_accuracy": settings["victim"]["test_accuracy"], "train_transform": "cifar_test",
        "temperature": 1.0, "environment_sha256": base.digest(ROOT / "uv.lock"),
    }
    if not summary["scan_complete"] or summary["evaluation"] != EVALUATION or summary["inputs"] != baseline_inputs or any(
        source_config[key] != value for key, value in expected.items()
    ) or any(source_config["settings"][key] != settings[key] for key in ("device", "num_workers", "cpu_threads")):
        raise ValueError("正式 Exp03 的来源配置与 Lab01 基线不一致。")
    if any(base.digest(ROOT / name) != digest for name, digest in source_config["code_sha256"].items()):
        raise ValueError("正式 Exp03 保存的实现标识与当前训练实现不一致。")
    saved_public = torch.load(weights / "public_init.pth", map_location="cpu", weights_only=True)
    if not same_state(public.state_dict(), saved_public["state_dict"]):
        raise ValueError("Lab01 的 B 公共初始化与正式 Exp03 不一致。")

    accuracy = read_rows(folder / "accuracy.tsv", RECORD_FIELDS, ("best_epoch", "protected_parameters"), "\t")
    history = read_rows(folder / "history.csv", exp03.HISTORY_FIELDS, ("epoch",))
    with (folder / "protection.tsv").open(newline="", encoding="utf-8") as stream:
        source_protection = {
            int(row["k"]) if row["k"] else None: row for row in csv.DictReader(stream, delimiter="\t")
        }
    records, selected_history = [], []
    for k, expected_mask in protection.items():
        group = group_key(expected_mask)
        matches = [row for row in accuracy if group_key(row) == group]
        group_history = [row for row in history if group_key(row) == group]
        if len(matches) != 1 or matches != [row for row in progress["completed"] if group_key(row) == group] or (
            group_history != [row for row in progress["history"] if group_key(row) == group]
        ) or source_protection[k]["protected_tensors"] != expected_mask["protected_tensors"]:
            raise ValueError(f"正式 Exp03 的原始产物与记录不符：{group}。")
        check_group(matches[0], group_history, training, expected_mask)
        records.extend(matches)
        selected_history.extend(group_history)
    if records[0]["best_accuracy"] != 17.71 or records[0]["best_accuracy"] != summary["all_shield_accuracy"]:
        raise ValueError("正式 Exp03 的 All-Shield Best 与协议不符。")

    source = {name: exp03.identity(folder / name) for name in (
        "accuracy.tsv", "history.csv", "protection.tsv", "summary.json",
    )}
    source["progress.pth"] = exp03.identity(weights / "progress.pth")
    source["public_init.pth"] = baseline_inputs["public_init"]
    names = {exp03.group_name(*group_key(row)) for row in records}
    checkpoints = [path for path in summary["checkpoints"] if Path(path).name.rsplit(".", 2)[0] in names]
    for name in checkpoints:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Exp03 摘要引用的权重不存在：{name}")
    return records, selected_history, source, checkpoints


def accuracy_rows(variant: str, records: list[dict], baseline: dict) -> list[dict]:
    blackbox = records[0]
    rows = []
    for record in records:
        gap = round(record["best_accuracy"] - blackbox["best_accuracy"], 6)
        final_gap = round(record["final_accuracy"] - blackbox["final_accuracy"], 6)
        ref = baseline.get(record["k"])
        rows.append({
            "variant": variant, **record, "gap_to_all_shield": gap, "final_gap_to_all_shield": final_gap,
            "delta_gap": round(gap - ref["gap_to_all_shield"], 6) if ref else 0.0,
            "final_delta_gap": round(final_gap - ref["final_gap_to_all_shield"], 6) if ref else 0.0,
            "source": "exp/03_topk" if variant == "B0" else str(LAB),
        })
    return rows


def summarize_variant(rows: list[dict], settings: dict) -> dict:
    blackbox, *joint = rows
    ks = [row["k"] for row in joint]
    scan_complete = ks == settings["k"]
    critical = next((row for row in joint if row["gap_to_all_shield"] <= settings["epsilon_pp"]), None) if scan_complete else None
    return {
        "scan_complete": scan_complete, "tested_k": ks,
        "all_shield_best_accuracy": blackbox["best_accuracy"],
        "all_shield_final_accuracy": blackbox["final_accuracy"],
        "threshold_best_accuracy": round(blackbox["best_accuracy"] + settings["epsilon_pp"], 6),
        "critical_k": critical["k"] if critical else None,
        "critical_protection": {key: critical[key] for key in ("protected_parameters", "protected_ratio")}
        if critical else None,
    }


def comparisons(by_variant: dict) -> dict:
    """逐个保护场景计算影响量及交互项，所有差值使用百分点。"""
    indexed = {variant: {row["k"]: row for row in rows} for variant, rows in by_variant.items()}

    def contrast(weights: dict) -> list[dict]:
        return [
            {"k": k, **{
                f"{metric}_pp": round(sum(weight * indexed[variant][k][metric]
                                          for variant, weight in weights.items()), 6)
                for metric in METRICS
            }}
            for k in indexed["B0"]
        ]

    return {
        "RQ1": {"head_B1_minus_B0": contrast({"B1": 1, "B0": -1})},
        "RQ2": {
            "lr_B2_minus_B0": contrast({"B2": 1, "B0": -1}),
            "wd_B3_minus_B0": contrast({"B3": 1, "B0": -1}),
            "lrwd_B4_minus_B0": contrast({"B4": 1, "B0": -1}),
            "lrwd_nonadditivity_B4_minus_B2_minus_B3_plus_B0": contrast({"B4": 1, "B2": -1, "B3": -1, "B0": 1}),
        },
        "RQ3": {
            "head_B1_minus_B0": contrast({"B1": 1, "B0": -1}),
            "head_B5_minus_B4": contrast({"B5": 1, "B4": -1}),
            "head_interaction_B5_minus_B4_minus_B1_plus_B0": contrast({"B5": 1, "B4": -1, "B1": -1, "B0": 1}),
        },
    }


def write_variant(folder: Path, rows: list[dict], history: list[dict], summary: dict) -> None:
    exp03.write_table(folder / "accuracy.tsv", rows, ACCURACY_FIELDS, delimiter="\t")
    exp03.write_table(folder / "history.csv", history, exp03.HISTORY_FIELDS)
    base.save_json(folder / "summary.json", summary)


def train_variant(variant_id, public, victim, rank, cached, evaluation, settings, b_config, run_config,
                  weights_dir, results_dir, protection, baseline, *, resume) -> tuple:
    training = run_config["training"]
    jobs = [group_key(row) for row in protection.values()]
    weights_dir.mkdir(parents=True, exist_ok=True)
    progress_path = weights_dir / "progress.pth"
    progress = {"configuration": run_config, "completed": [], "history": []}
    if progress_path.exists():
        if not resume:
            raise FileExistsError(f"已有进度，请使用 --resume：{progress_path}")
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress["configuration"] != run_config:
            raise ValueError(f"{variant_id} 的实际配置、输入或实现与已保存进度不一致。")
    completed, history = progress["completed"], progress["history"]
    epochs = training["epochs"]
    if [group_key(row) for row in completed] != jobs[:len(completed)] or len(history) != len(completed) * epochs:
        raise ValueError(f"{variant_id} 的已完成训练组或历史不完整。")
    for i, record in enumerate(completed):
        check_group(record, history[i * epochs:(i + 1) * epochs], training, protection[record["k"]])

    base.save_checkpoint(progress_path, progress)
    exp03.write_table(results_dir / "history.csv", history, exp03.HISTORY_FIELDS)
    for strategy, k in jobs[len(completed):]:
        print(f"{variant_id} [{len(completed) + 1}/{len(jobs)}] {exp03.group_name(strategy, k)}", flush=True)
        record, group_history = exp03.train_group(
            public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
            weights_dir, results_dir, strategy, k, previous_history=history,
        )
        check_group(record, group_history, training, protection[k])
        completed.append(record)
        history.extend(group_history)
        base.save_checkpoint(progress_path, progress)
        exp03.write_table(results_dir / "accuracy.tsv", accuracy_rows(variant_id, completed, baseline),
                          ACCURACY_FIELDS, delimiter="\t")

    rows = accuracy_rows(variant_id, completed, baseline)
    summary = summarize_variant(rows, settings)
    keep = {k for k in (None, summary["critical_k"])}
    checkpoints = {}
    for strategy, k in jobs:
        checkpoints[k] = [
            weights_dir / f"{exp03.group_name(strategy, k)}.{kind}.pth" for kind in ("best", "last")
        ]
    retained = [path for k, paths in checkpoints.items() if k in keep for path in paths]
    for path in retained:
        if not path.is_file():
            raise FileNotFoundError(f"{variant_id} 的应保留权重缺失：{path}")
    summary["checkpoints"] = [str(path.relative_to(ROOT)) for path in retained]
    write_variant(results_dir, rows, history, summary)
    for k, paths in checkpoints.items():
        if k not in keep:
            for path in paths:
                path.unlink(missing_ok=True)
    return rows, history, summary


def run_experiment(settings, variants, training, b_config, victim, rank, cached, publics, protection, inputs,
                   baseline_data, results_dir, weights_dir, *, resume) -> None:
    device = torch.device(settings["device"])
    evaluation = exp03.load_evaluation(settings, device)
    victim_accuracy = exp03.evaluate(victim.to(device), *evaluation, training["batch_size"])
    victim.cpu()
    if victim_accuracy != settings["victim"]["test_accuracy"]:
        raise ValueError("当前评测未复现正式 Victim 的测试准确率。")

    records, baseline_history, baseline_source, baseline_checkpoints = baseline_data
    baseline_rows = accuracy_rows("B0", records, {})
    baseline = {row["k"]: row for row in baseline_rows}
    baseline_summary = {
        **summarize_variant(baseline_rows, settings), "source": baseline_source, "checkpoints": baseline_checkpoints,
    }
    write_variant(results_dir / "B0", baseline_rows, baseline_history, baseline_summary)
    code = {name: base.digest(ROOT / name) for name in (
        "config.py", "randomness.py", "models/build.py", "models/weights.py",
        "exp/01_train_base/train.py", "exp/03_topk/train.py", "lab/01_basecheck/train.py",
    )}
    run_config = {
        "settings": settings, "deterministic": b_config.deterministic, "b_ratio": b_config.ratio_name,
        "rank_field": exp03.RANK_FIELD, "fixed_protected": sorted(exp03.FIXED_PROTECTED),
        "inputs": {**inputs, "public_init_b": baseline_source["public_init.pth"],
                   "official_test": exp03.identity(ROOT / "dataset/raw/c100/cifar-100-python/test")},
        "baseline_source": baseline_source, "config": exp03.identity(HERE / "config.toml"),
        "environment_sha256": base.digest(ROOT / "uv.lock"), "code_sha256": code,
        "train_transform": "cifar_test", "temperature": 1.0,
    }
    by_variant, summaries, all_history = {"B0": baseline_rows}, {"B0": baseline_summary}, []
    for variant_id in ("B1", "B2", "B3", "B4", "B5"):
        variant = variants[variant_id]
        effective = {
            **run_config, "variant": variant_id, "head": variant["head"],
            "head_seed": settings["a_seed"] if variant["head"] == "A" else exp03.head_seed(settings, b_config),
            "training": {**training, "lr": variant["lr"], "weight_decay": variant["weight_decay"]},
        }
        rows, history, summary = train_variant(
            variant_id, publics[variant["head"]], victim, rank, cached, evaluation, settings, b_config, effective,
            weights_dir / variant_id, results_dir / variant_id, protection, baseline, resume=resume,
        )
        by_variant[variant_id] = rows
        summaries[variant_id] = summary
        all_history.extend({"variant": variant_id, **row} for row in history)

    all_accuracy = [row for rows in by_variant.values() for row in rows]
    if len(all_accuracy) != 54 or len(all_history) != 4500 or not all(
        summary["scan_complete"] for summary in summaries.values()
    ):
        raise ValueError("Lab01 汇总缺少完整训练结果。")
    for source in (*run_config["inputs"].values(), *baseline_source.values(), run_config["config"]):
        if base.digest(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"训练期间实验输入发生变化：{source['path']}")
    exp03.write_table(results_dir / "accuracy.tsv", all_accuracy, ACCURACY_FIELDS, delimiter="\t")
    exp03.write_table(results_dir / "history.csv", all_history, HISTORY_FIELDS)
    exp03.write_table(results_dir / "protection.tsv", list(protection.values()), exp03.PROTECTION_FIELDS, delimiter="\t")
    base.save_json(results_dir / "summary.json", {
        "evaluation": EVALUATION, "victim_accuracy": victim_accuracy, "epsilon_pp": settings["epsilon_pp"],
        "variants": summaries, "comparisons": comparisons(by_variant),
    })
    plot = importlib.import_module("lab.01_basecheck.plot")
    plot.plot_results(results_dir)
    print(f"训练与汇总完成：{results_dir.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="只检查配置、正式来源和初始化")
    modes.add_argument("--resume", action="store_true", help="跳过已完成组，从头执行中断的组")
    args = parser.parse_args()
    settings, variants, training = read_settings()
    a_config, b_config, query_config = (load_config(seed=settings[key]) for key in ("a_seed", "b_seed", "query_seed"))
    seed_everything(b_config)
    torch.set_num_threads(settings["cpu_threads"])

    victim, rank, cached, inputs = load_inputs(settings, a_config, query_config)
    publics = build_publics(settings, a_config, b_config)
    protection = lab_protection(publics["B"], rank, settings["k"])
    check_initialization(publics, victim, protection)
    baseline = load_baseline(settings, b_config, training, publics["B"], inputs, protection)
    print(f"B0 已验证 {len(baseline[0])} 组正式结果；B1–B5 共 45 组，每组 {training['epochs']} 轮。", flush=True)
    if args.dry_run:
        return

    if torch.device(settings["device"]).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    results = ROOT / "results" / LAB / run_path(settings)
    weights = ROOT / "weights" / LAB / run_path(settings)
    weights.mkdir(parents=True, exist_ok=True)
    # 锁定运行目录，避免并行进程修改同一份组进度或清理同一份权重。
    directory = os.open(weights, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"已有进程正在执行 Lab01：{weights}") from error
        if not args.resume and (any(weights.iterdir()) or results.exists() and any(results.iterdir())):
            raise FileExistsError("已有 Lab01 产物，请使用 --resume。")
        run_experiment(settings, variants, training, b_config, victim, rank, cached, publics, protection, inputs,
                       baseline, results, weights, resume=args.resume)
    finally:
        os.close(directory)


if __name__ == "__main__":
    main()
