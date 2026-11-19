"""按作者 Rank 和固定 bias 约定，复现 TensorShield 的标准 MS 保护效果。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import tomli
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from config import load_config
from models.build import build_model
from models.weights import WEIGHTS
from randomness import preserve_rng_state, seed_everything, seed_worker, torch_generator

# 仅复用已有变换和文件保存工具，B 独立配置训练与选模。
base = importlib.import_module("exp.01_train_base.train")
OUTPUT_SUBDIR = "author_ms_fixed_bias"
RANK_FIELD = "author_rank"
FIXED_PROTECTED = {"last_linear.bias"}
HISTORY_FIELDS = ("strategy", "k", "epoch", "lr", "train_loss", "test_accuracy", "seconds")
ACCURACY_FIELDS = ("strategy", "k", "initial_accuracy", "best_accuracy", "best_epoch", "final_accuracy", "protected_parameters",
                   "protected_ratio", "gap_to_all_shield")
PROTECTION_FIELDS = ("strategy", "k", "head_mode", "protected_parameters", "protected_ratio", "protected_tensors")


def identity(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": base.digest(path)}


def write_table(path: Path, rows: list[dict], fields, delimiter=",") -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(stream.getvalue(), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_settings(path: Path) -> dict:
    with path.open("rb") as stream:
        settings = tomli.load(stream)
    if (settings["model"], settings["dataset"]) != ("resnet18", "c100"):
        raise ValueError("本轮协议限定为 resnet18+c100。")
    for key in ("a_seed", "query_seed"):
        load_config(seed=settings[key])
    if not settings["b_seeds"] or len(set(settings["b_seeds"])) != len(settings["b_seeds"]):
        raise ValueError("B 重复种子必须非空且不重复。")
    for seed in settings["b_seeds"]:
        load_config(seed=seed)
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("num_workers 必须是非负整数。")
    if type(settings["cpu_threads"]) is not int or settings["cpu_threads"] < 1:
        raise ValueError("cpu_threads 必须是正整数。")
    if not math.isfinite(settings["epsilon_pp"]) or settings["epsilon_pp"] < 0:
        raise ValueError("黑盒容差必须是有限非负数。")
    probe = settings["probe_k"]
    if not probe or len(set(probe)) != len(probe) or any(type(k) is not int or k < 1 for k in probe):
        raise ValueError("首阶段的 k 必须是不重复的正整数。")
    return settings


def read_training(path: Path) -> dict:
    """读取独立的 B 训练配置，拒绝误用 A 的真实标签交叉熵配置。"""
    with path.open("rb") as stream:
        training = tomli.load(stream)
    expected = {"epochs", "batch_size", "loss", "optimizer", "lr", "momentum", "weight_decay", "scheduler", "step_size", "gamma"}
    if set(training) != expected:
        raise ValueError(f"B 训练配置必须包含且仅包含：{', '.join(sorted(expected))}")
    for key in ("epochs", "batch_size", "step_size"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} 必须是正整数。")
    for key in ("lr", "momentum", "weight_decay", "gamma"):
        if type(training[key]) not in (int, float) or not math.isfinite(training[key]):
            raise ValueError(f"{key} 必须是有限数值。")
    if not (training["lr"] > 0 and 0 <= training["momentum"] < 1 and training["weight_decay"] >= 0 and 0 < training["gamma"] <= 1):
        raise ValueError("B 的优化器参数超出允许范围。")
    if (training["loss"], training["optimizer"], training["scheduler"]) != ("soft_cross_entropy", "sgd", "step"):
        raise ValueError("B 使用 soft_cross_entropy、sgd 和 step。")
    return training


def head_seed(settings: dict, b_config) -> int:
    """以独立命名空间派生 B 分类头随机流，不依赖 A 种子或 Python 哈希。"""
    key = f"alchemy/03_topk/b_head/{settings['model']}/{settings['dataset']}/{b_config.seed}"
    # 最高位置 1，使分类头种子与全局允许的 32 位训练种子范围分离。
    return (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & (2**63 - 1)) | 2**62


def build_public(settings: dict, b_config) -> nn.Module:
    """加载公共主干，再独立重置 B 分类头；不消耗训练随机状态。"""
    public = build_model(settings["model"], settings["dataset"], b_config).eval()
    with preserve_rng_state():
        torch.random.default_generator.manual_seed(head_seed(settings, b_config))
        public.last_linear.reset_parameters()
    return public


def load_victim(settings: dict, a_config) -> tuple:
    """核对并加载已训练 Victim；保持已有 checkpoint 只读。"""
    relative = Path(settings["model"]) / settings["dataset"] / f"seed_{a_config.seed}"
    path = ROOT / "weights/01_train_base" / relative / "best.pth"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved.get("config", saved.get("metadata", {}))
    for key, expected in (("model", settings["model"]), ("dataset", settings["dataset"]), ("seed", a_config.seed)):
        if metadata.get(key) != expected:
            raise ValueError(f"Victim 的 {key} 与本次配置不符。")
    if saved["epoch"] != settings["victim"]["epoch"] or not math.isclose(
        saved["selection"]["accuracy"], settings["victim"]["test_accuracy"], abs_tol=1e-9
    ):
        raise ValueError("Victim 的轮次或准确率与协议不符。")
    if metadata["pretrained_sha256"] != WEIGHTS[settings["model"]].sha256:
        raise ValueError("公共预训练权重与 Victim 的训练起点不一致。")
    for name in ("config.py", "randomness.py", "models/build.py", "models/weights.py"):
        expected = metadata.get("code_sha256", {}).get(name)
        if expected is not None and base.digest(ROOT / name) != expected:
            raise ValueError(f"初始化实现与 A 保存的记录不同：{name}")
    victim = build_model(settings["model"], settings["dataset"], a_config).eval()
    if victim.state_dict().keys() != saved["state_dict"].keys() or any(
        value.shape != saved["state_dict"][name].shape for name, value in victim.state_dict().items()
    ):
        raise ValueError("公共模型与 Victim 的参数名称或形状不一致。")
    victim.load_state_dict(saved["state_dict"], strict=True)
    return victim, path


def load_rank(path: Path, model: nn.Module) -> list[str]:
    """动态读取候选排名，核对完整候选集合、连续编号和参数大小。"""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows.sort(key=lambda row: int(row[RANK_FIELD]))
    names = [row["tensor"] for row in rows]
    expected = {f"{name}.weight" for name, module in model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Linear)) and name != "conv1" and "downsample" not in name}
    if not rows or len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Rank 必须无重复、无遗漏地覆盖 TensorShield 的候选参数。")
    if [int(row[RANK_FIELD]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("author_rank 编号必须从 1 连续递增。")
    parameters = dict(model.named_parameters())
    for row in rows:
        if int(row["parameters"]) != parameters[row["tensor"]].numel():
            raise ValueError(f"Rank 中的参数数量不匹配：{row['tensor']}")
    return names


def protected_names(model: nn.Module, rank: list[str], k: int) -> set[str]:
    """每个 Top-k 固定保护分类头 bias；bias 不占排名单位。"""
    if not 1 <= k <= len(rank):
        raise ValueError("k 超出候选范围。")
    if not FIXED_PROTECTED <= dict(model.named_parameters()).keys():
        raise ValueError("模型缺少需要固定保护的分类头 bias。")
    return set(rank[:k]) | FIXED_PROTECTED


def make_student(public: nn.Module, victim: nn.Module, protected: set[str], strategy: str) -> nn.Module:
    """受保护参数取公共初始化，其余参数及所有 BN buffers 取 Victim。"""
    if strategy not in {"all_shield", "freeze", "joint"}:
        raise ValueError("未知训练策略。")
    student = copy.deepcopy(public)
    if strategy != "all_shield":
        initial, exposed = public.state_dict(), victim.state_dict()
        student.load_state_dict({name: initial[name] if name in protected else value
                                 for name, value in exposed.items()}, strict=True)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(strategy != "freeze" or name in protected)
    return student


def fix_public_initialization(public, settings, b_config, path: Path) -> Path:
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved.get("state_dict", saved)
        if state.keys() != public.state_dict().keys() or any(
            not torch.equal(value, state[name]) for name, value in public.state_dict().items()
        ):
            raise ValueError("已有 B 公共初始化与当前独立随机流不一致。")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(path, {"state_dict": public.state_dict(), "model": settings["model"],
                                   "dataset": settings["dataset"], "b_seed": b_config.seed,
                                   "head_seed": head_seed(settings, b_config),
                                   "pretrained_sha256": WEIGHTS[settings["model"]].sha256})
    return path


def read_indices(path: Path, ratio) -> torch.Tensor:
    values = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != math.floor(50000 * ratio) or not values or len(values) != len(set(values)):
        raise ValueError("查询索引的数量或唯一性不符合固定比例。")
    if min(values) < 0 or max(values) >= 50000:
        raise ValueError("查询索引超出 CIFAR-100 官方训练集。")
    return torch.tensor(values, dtype=torch.int64)


def load_queries(path, indices, indices_path, victim, victim_path, settings) -> dict:
    """只读复用已有的查询图片与完整概率，所有样本直接进入 B 训练。"""
    _, transform = base.build_transforms(settings["dataset"])
    raw = ROOT / "dataset/raw" / settings["dataset"]
    metadata = {"victim_sha256": base.digest(victim_path), "indices_sha256": base.digest(indices_path),
                "train_sha256": base.digest(raw / "cifar-100-python/train"),
                "transform": repr(transform), "temperature": 1.0, "classes": victim.last_linear.out_features}
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if cached["metadata"] != metadata or not torch.equal(cached["indices"], indices):
        raise ValueError("已有查询缓存与 Victim、查询索引或输入处理不一致。")
    images, probabilities = cached["images"], cached["probabilities"]
    if images.shape != (len(indices), 3, 32, 32) or images.dtype != torch.float32 or not torch.isfinite(images).all():
        raise ValueError("查询图片缓存形状、精度或数值无效。")
    if probabilities.shape != (len(indices), metadata["classes"]) or probabilities.dtype != torch.float32:
        raise ValueError("查询软标签的形状或精度无效。")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(dim=1), torch.ones(len(indices)), atol=1e-6, rtol=0
    ):
        raise ValueError("查询软标签必须是有效的完整概率分布。")
    print(f"查询缓存：{len(indices)} 张固定输入、{metadata['classes']} 类完整概率。", flush=True)
    return cached


def load_evaluation(settings, device) -> tuple[torch.Tensor, torch.Tensor]:
    """确定性测试变换只执行一次，评测仍按训练配置中的批次大小进行。"""
    _, transform = base.build_transforms(settings["dataset"])
    dataset = datasets.CIFAR100(ROOT / "dataset/raw" / settings["dataset"], train=False, download=False)
    images = torch.stack([transform(Image.fromarray(image)) for image in dataset.data]).to(device)
    labels = torch.tensor(dataset.targets, dtype=torch.int64, device=device)
    return images, labels


def evaluate(model, images, labels, batch_size) -> float:
    """在全部官方测试图片上评测，不消耗训练随机状态；选模由训练循环处理。"""
    was_training = model.training
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device=labels.device)
    finite = torch.ones((), dtype=torch.bool, device=labels.device)
    try:
        with preserve_rng_state(), torch.inference_mode():
            for start in range(0, len(labels), batch_size):
                logits = model(images[start:start + batch_size])
                finite.logical_and_(torch.isfinite(logits).all())
                correct.add_(logits.argmax(dim=1).eq(labels[start:start + batch_size]).sum())
    finally:
        model.train(was_training)
    if not finite.item():
        raise RuntimeError("评测输出包含非有限值。")
    return 100 * correct.item() / len(labels)


def train_epoch(model, loader, optimizer, device) -> float:
    """所有模块保持 train 模式；冻结仅由 requires_grad 和优化器参数列表实现。"""
    model.train()
    total, count = 0.0, 0
    for images, probabilities in loader:
        images, probabilities = images.to(device, non_blocking=True), probabilities.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = -(probabilities * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("软标签训练出现非有限损失。")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
        count += len(images)
    return total / count


def group_name(strategy, k) -> str:
    return strategy if k is None else f"{strategy}_top_{k:02d}"


def accuracy_rows(completed: list[dict]) -> list[dict]:
    if not completed:
        return []
    blackbox = completed[0]["best_accuracy"]
    return [{**row, "gap_to_all_shield": round(row["best_accuracy"] - blackbox, 6)} for row in completed]


def summarize(completed, victim_accuracy, epsilon, inputs, rank_count) -> dict:
    """使用 Best Accuracy 比较；首阶段仅报告观测值，不推断全扫描的最小 k。"""
    blackbox = completed[0]["best_accuracy"]
    threshold = blackbox + epsilon
    by_strategy = {strategy: {row["k"]: row for row in completed if row["strategy"] == strategy}
                   for strategy in ("freeze", "joint")}
    expected = set(range(1, rank_count + 1))
    scan_complete = all(set(rows) == expected for rows in by_strategy.values())
    observed = {}
    for strategy, rows in by_strategy.items():
        observed[strategy] = min((k for k, row in rows.items() if row["best_accuracy"] <= threshold), default=None)
    common = set(by_strategy["freeze"]) & set(by_strategy["joint"])
    observed["both"] = min((k for k in common
                            if all(rows[k]["best_accuracy"] <= threshold for rows in by_strategy.values())), default=None)
    critical = observed if scan_complete else dict.fromkeys(observed)
    protection = {}
    for strategy, k in critical.items():
        if k is None:
            protection[strategy] = None
        else:
            row = by_strategy["freeze" if strategy == "both" else strategy][k]
            protection[strategy] = {key: row[key] for key in ("protected_parameters", "protected_ratio")}
    return {"victim_accuracy": victim_accuracy,
            "all_shield_accuracy": blackbox, "all_shield_final_accuracy": completed[0]["final_accuracy"], "epsilon_pp": epsilon,
            "threshold_accuracy": threshold, "critical_k": critical, "critical_protection": protection,
            "scan_complete": scan_complete, "candidate_count": rank_count,
            "tested_k": {strategy: sorted(rows) for strategy, rows in by_strategy.items()},
            "observed_min_k": observed, "inputs": inputs,
            "evaluation": "official_test_used_for_best_epoch_and_topk_selection"}


def planned_jobs(rank: list[str], settings: dict, stage: str) -> list[tuple]:
    """先运行黑盒与关键 k；完整扫描在同一进度后补齐其余候选。"""
    if stage not in {"probe", "full"} or any(k > len(rank) for k in settings["probe_k"]):
        raise ValueError("运行阶段或首阶段 k 超出候选范围。")
    ks = list(settings["probe_k"])
    if stage == "full":
        ks += [k for k in range(1, len(rank) + 1) if k not in ks]
    return [("all_shield", None)] + [(strategy, k) for k in ks for strategy in ("freeze", "joint")]


def protection_rows(public, rank) -> list[dict]:
    """逐项列出真实保护参数；两种策略共用 Top-k 集合，BN buffers 不计参数量。"""
    parameters = dict(public.named_parameters())
    total = sum(parameter.numel() for parameter in parameters.values())
    rows = []
    for k in [None, *range(1, len(rank) + 1)]:
        protected = set(parameters) if k is None else protected_names(public, rank, k)
        count = sum(parameters[name].numel() for name in protected)
        rows.append({"strategy": "all_shield" if k is None else "freeze/joint", "k": k,
                     "head_mode": "replace" if "last_linear.weight" in protected else "mixed",
                     "protected_parameters": count, "protected_ratio": count / total,
                     "protected_tensors": ";".join(sorted(protected))})
    return rows


def train_group(public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
                weights_dir, results_dir, strategy, k, previous_history=()) -> tuple[dict, list]:
    """完整训练一组，按测试准确率选 Best；同分保留较晚轮次，同时保存末轮。"""
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(b_config)
    device = torch.device(settings["device"])
    parameters = dict(public.named_parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters.values())
    protected = set(parameters) if strategy == "all_shield" else protected_names(public, rank, k)
    protected_count = sum(parameters[name].numel() for name in protected)
    model = make_student(public, victim, protected, strategy).to(device)
    dataset = TensorDataset(cached["images"], cached["probabilities"])
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        num_workers=settings["num_workers"], worker_init_fn=seed_worker,
                        generator=torch_generator(b_config), pin_memory=device.type == "cuda",
                        drop_last=False, persistent_workers=False)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad],
                                lr=training["lr"], momentum=training["momentum"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["step_size"], gamma=training["gamma"])
    initial = evaluate(model, *evaluation, training["batch_size"])
    group = group_name(strategy, k)
    print(f"{group}：初始准确率 {initial:.2f}%，保护参数 {protected_count}/{total_parameters}。", flush=True)
    group_history = []
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, training["epochs"] + 1):
        started = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, loader, optimizer, device)
        accuracy = evaluate(model, *evaluation, training["batch_size"])
        if accuracy >= best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        scheduler.step()
        group_history.append({"strategy": strategy, "k": k, "epoch": epoch, "lr": lr,
                              "train_loss": loss, "test_accuracy": accuracy, "seconds": time.monotonic() - started})
        if epoch == 1 or epoch % 10 == 0 or epoch == training["epochs"]:
            write_table(results_dir / "history.csv", list(previous_history) + group_history, HISTORY_FIELDS)
            print(f"  {group} {epoch:03d}/{training['epochs']}：loss={loss:.6f}，test={accuracy:.2f}%，"
                  f"best={best_accuracy:.2f}%@{best_epoch}", flush=True)
    record = {"strategy": strategy, "k": k, "initial_accuracy": initial,
              "best_accuracy": best_accuracy, "best_epoch": best_epoch, "final_accuracy": accuracy,
              "protected_parameters": protected_count, "protected_ratio": protected_count / total_parameters}
    metadata = {"config": run_config, "protected": sorted(protected), **record}
    base.save_checkpoint(weights_dir / f"{group}.best.pth", {
        **metadata, "state_dict": best_state, "epoch": best_epoch, "selection": "best_test_accuracy_later_tie"})
    base.save_checkpoint(weights_dir / f"{group}.last.pth", {
        **metadata, "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": training["epochs"], "selection": "last_epoch"})
    return record, group_history


def run_scan(public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
             weights_dir, results_dir, *, stage="probe", resume=False) -> dict:
    """按完整训练组恢复；首阶段结束即返回，完整扫描需要显式选择 full。"""
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = weights_dir / "progress.pth"
    jobs = planned_jobs(rank, settings, stage)
    progress = {"configuration": run_config, "completed": [], "history": []}
    if progress_path.exists():
        if not resume:
            raise FileExistsError("已有本次训练进度，请使用 --resume。")
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress["configuration"] != run_config:
            raise ValueError("恢复训练的输入、配置或代码与已保存进度不一致。")
    completed, history = progress["completed"], progress["history"]
    if [(row["strategy"], row["k"]) for row in completed] != jobs[:len(completed)]:
        raise ValueError("已完成训练组不符合当前阶段的执行顺序；完整扫描使用 --stage full。")
    if len(history) != len(completed) * training["epochs"]:
        raise ValueError("已完成训练组的逐轮记录不完整。")
    base.save_checkpoint(progress_path, progress)
    write_table(results_dir / "history.csv", history, HISTORY_FIELDS)
    write_table(results_dir / "protection.tsv", protection_rows(public, rank), PROTECTION_FIELDS, delimiter="\t")
    for strategy, k in jobs[len(completed):]:
        print(f"[{len(completed) + 1}/{len(jobs)}]", flush=True)
        record, group_history = train_group(
            public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
            weights_dir, results_dir, strategy, k, previous_history=history)
        completed.append(record)
        history.extend(group_history)
        base.save_checkpoint(progress_path, progress)
        write_table(results_dir / "history.csv", history, HISTORY_FIELDS)
        write_table(results_dir / "accuracy.tsv", accuracy_rows(completed), ACCURACY_FIELDS, delimiter="\t")
    summary = summarize(completed, run_config["victim_accuracy"], settings["epsilon_pp"], run_config["inputs"], len(rank))
    # 首阶段保留所有权重，完整扫描结束后仅保留黑盒与最终临界组的 Best / Last。
    keep = {group_name(strategy, k) for strategy, k in jobs} if not summary["scan_complete"] else {"all_shield"}
    for strategy, k in summary["critical_k"].items():
        if k is not None:
            keep.update(group_name(item, k) for item in (("freeze", "joint") if strategy == "both" else (strategy,)))
    for strategy, k in jobs:
        name = group_name(strategy, k)
        for suffix in ("best", "last"):
            path = weights_dir / f"{name}.{suffix}.pth"
            if name in keep:
                if not path.is_file():
                    raise FileNotFoundError(f"已完成组缺少需保留的权重：{path}")
            else:
                path.unlink(missing_ok=True)
    summary["checkpoints"] = [str((weights_dir / f"{name}.{suffix}.pth").relative_to(ROOT))
                              for name in sorted(keep) for suffix in ("best", "last")]
    base.save_json(results_dir / "summary.json", summary)
    write_table(results_dir / "accuracy.tsv", accuracy_rows(completed), ACCURACY_FIELDS, delimiter="\t")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--b-seed", type=int)
    parser.add_argument("--all-seeds", action="store_true", help="运行本实验声明的全部 B 重复种子")
    parser.add_argument("--stage", choices=("probe", "full"), default="probe", help="probe 只运行黑盒与关键 k；full 补齐全部 Top-k")
    parser.add_argument("--resume", action="store_true", help="跳过已完成训练组，重新执行中断的组")
    parser.add_argument("--dry-run", action="store_true", help="只核对模型、Rank、查询索引和配置")
    args = parser.parse_args()
    if args.all_seeds and args.b_seed is not None:
        parser.error("--all-seeds 和 --b-seed 不能同时使用。")
    settings = read_settings(args.config)
    training_path = ROOT / settings["training_config"]
    training = read_training(training_path)
    a_config, query_config = load_config(seed=settings["a_seed"]), load_config(seed=settings["query_seed"])
    b_seeds = settings["b_seeds"] if args.all_seeds else [load_config(seed=args.b_seed).seed]
    seed_everything(a_config)
    torch.set_num_threads(settings["cpu_threads"])
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    victim, victim_path = load_victim(settings, a_config)
    rank_path = ROOT / "results/02_rank" / settings["model"] / settings["dataset"] / f"seed_{a_config.seed}/rank.tsv"
    rank = load_rank(rank_path, victim)
    indices_path = ROOT / "dataset/splits" / settings["dataset"] / f"seed_{query_config.seed}" / f"ratio_{query_config.ratio_name}/b_indices.txt"
    indices = read_indices(indices_path, query_config.b_ratio)
    if len(indices) != 500:
        raise ValueError("本实验固定复用已有 500 张查询图片。")
    relative = Path(settings["model"]) / settings["dataset"] / f"a_seed_{a_config.seed}"
    jobs = planned_jobs(rank, settings, args.stage)
    query_path = ROOT / "dataset/queries" / relative / f"query_seed_{query_config.seed}" / f"ratio_{query_config.ratio_name}/probabilities.pth"
    cached = load_queries(query_path, indices, indices_path, victim, victim_path, settings)
    print(f"按 {RANK_FIELD} 读取 {len(rank)} 项候选；阶段 {args.stage}，每个 B seed {len(jobs)} 组，"
          f"每组 {training['epochs']} 轮；固定查询 {len(indices)} 张，无训练增强。", flush=True)
    if args.dry_run:
        for seed in b_seeds:
            public = build_public(settings, load_config(seed=seed))
            rows = protection_rows(public, rank)
            print(f"B seed={seed}，独立 head seed={head_seed(settings, load_config(seed=seed))}；"
                  f"Top-1 / Top-2 保护参数={[row['protected_parameters'] for row in rows[1:3]]}。", flush=True)
        return
    for seed in b_seeds:
        for parent in ("weights", "results"):
            path = ROOT / parent / "03_topk" / OUTPUT_SUBDIR / relative / f"b_seed_{seed}"
            if path.exists() and any(path.iterdir()) and not args.resume:
                raise FileExistsError(f"已有实验产物，请使用 --resume：{path}")
    evaluation = load_evaluation(settings, device)
    victim_accuracy = evaluate(victim.to(device), *evaluation, training["batch_size"])
    victim.cpu()
    if not math.isclose(victim_accuracy, settings["victim"]["test_accuracy"], abs_tol=1e-9):
        raise ValueError("当前评测未复现 Victim 的准确率。")
    inputs = {name: identity(path) for name, path in {
        "rank": rank_path, "victim": victim_path,
        "query_indices": indices_path, "query_cache": query_path}.items()}
    code = {name: base.digest(ROOT / name) for name in (
        "config.py", "randomness.py", "models/build.py", "models/weights.py",
        "exp/01_train_base/train.py", "exp/03_topk/train.py")}
    for seed in b_seeds:
        b_config = load_config(seed=seed)
        public = build_public(settings, b_config)
        destination = relative / f"b_seed_{seed}"
        weights_dir, results_dir = (ROOT / parent / "03_topk" / OUTPUT_SUBDIR / destination for parent in ("weights", "results"))
        public_path = fix_public_initialization(public, settings, b_config, weights_dir / "public_init.pth")
        run_inputs = {**inputs, "public_init": identity(public_path)}
        run_config = {"settings": settings, "training": training, "head_seed": head_seed(settings, b_config),
                      "rank_field": RANK_FIELD, "fixed_protected": sorted(FIXED_PROTECTED),
                      "a_seed": a_config.seed, "query_seed": query_config.seed, "b_seed": seed,
                      "b_ratio": query_config.ratio_name, "deterministic": b_config.deterministic,
                      "inputs": run_inputs, "training_config_sha256": base.digest(training_path),
                      "environment_sha256": base.digest(ROOT / "uv.lock"), "code_sha256": code,
                      "victim_accuracy": victim_accuracy, "train_transform": "cifar_test", "temperature": 1.0}
        summary = run_scan(public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
                           weights_dir, results_dir, stage=args.stage, resume=args.resume)
        from plot import plot_results
        plot_results(results_dir)
        print(json.dumps({"b_seed": seed, "all_shield_accuracy": summary["all_shield_accuracy"],
                          "scan_complete": summary["scan_complete"], "critical_k": summary["critical_k"],
                          "observed_min_k": summary["observed_min_k"]}, ensure_ascii=False), flush=True)
        if base.digest(public_path) != run_inputs["public_init"]["sha256"]:
            raise RuntimeError("B 公共初始化在训练期间被改写。")
    for value in inputs.values():
        if base.digest(ROOT / value["path"]) != value["sha256"]:
            raise RuntimeError(f"实验输入文件被改写：{value['path']}")


if __name__ == "__main__":
    main()
