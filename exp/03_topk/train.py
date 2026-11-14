"""扫描正式 Rank 的全部 Top-k，使用同一查询缓存训练 Freeze、Joint 和 All-Shield。"""

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

# 仅复用已有变换、配置校验和文件保存工具，不调用 A 的测试选模训练循环。
base = importlib.import_module("exp.01_train_base.train")
HISTORY_FIELDS = ("strategy", "k", "epoch", "lr", "train_loss", "test_accuracy", "seconds")
ACCURACY_FIELDS = ("strategy", "k", "initial_accuracy", "final_accuracy", "protected_parameters",
                   "protected_ratio", "gap_to_all_shield")


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
    return settings


def load_models(settings: dict, a_config) -> tuple:
    """核对 Victim 的来源，并构造 A 实际训练起点；保持已有 checkpoint 只读。"""
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
    public = build_model(settings["model"], settings["dataset"], a_config).eval()
    victim = copy.deepcopy(public)
    if public.state_dict().keys() != saved["state_dict"].keys() or any(
        value.shape != saved["state_dict"][name].shape for name, value in public.state_dict().items()
    ):
        raise ValueError("公共模型与 Victim 的参数名称或形状不一致。")
    victim.load_state_dict(saved["state_dict"], strict=True)
    return public, victim, path


def load_rank(path: Path, model: nn.Module) -> list[str]:
    """动态读取候选排名，核对完整候选集合、连续编号和参数大小。"""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows.sort(key=lambda row: int(row["rank"]))
    names = [row["tensor"] for row in rows]
    expected = {f"{name}.weight" for name, module in model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Linear)) and name != "conv1" and "downsample" not in name}
    if not rows or len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Rank 必须无重复、无遗漏地覆盖 TensorShield 的候选参数。")
    if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Rank 编号必须从 1 连续递增。")
    parameters = dict(model.named_parameters())
    for row in rows:
        if int(row["parameters"]) != parameters[row["tensor"]].numel():
            raise ValueError(f"Rank 中的参数数量不匹配：{row['tensor']}")
    return names


def protected_names(model: nn.Module, rank: list[str], k: int) -> set[str]:
    """分类头的 weight 和 bias 作为同一个排序单位进入保护集合。"""
    if not 1 <= k <= len(rank):
        raise ValueError("k 超出候选范围。")
    protected = set(rank[:k])
    if "last_linear.weight" in protected and model.last_linear.bias is not None:
        protected.add("last_linear.bias")
    return protected


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


def fix_public_initialization(public, settings, a_config) -> Path:
    path = ROOT / "weights/public_init" / settings["model"] / settings["dataset"] / f"seed_{a_config.seed}/init.pth"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved.get("state_dict", saved)
        if state.keys() != public.state_dict().keys() or any(
            not torch.equal(value, state[name]) for name, value in public.state_dict().items()
        ):
            raise ValueError("已有公共初始化与 A 的实际训练起点不一致。")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(path, {"state_dict": public.state_dict(), "model": settings["model"],
                                   "dataset": settings["dataset"], "seed": a_config.seed,
                                   "pretrained_sha256": WEIGHTS[settings["model"]].sha256})
    return path


def read_indices(path: Path, ratio) -> torch.Tensor:
    values = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != math.floor(50000 * ratio) or not values or len(values) != len(set(values)):
        raise ValueError("查询索引的数量或唯一性不符合固定比例。")
    if min(values) < 0 or max(values) >= 50000:
        raise ValueError("查询索引超出 CIFAR-100 官方训练集。")
    return torch.tensor(values, dtype=torch.int64)


def prepare_queries(path, indices, indices_path, victim, victim_path, settings, batch_size, device) -> dict:
    """缓存实际查询的图片张量与完整概率，B 直接训练这些输入，不读取真实标签。"""
    _, transform = base.build_transforms(settings["dataset"])
    raw = ROOT / "dataset/raw" / settings["dataset"]
    metadata = {"victim_sha256": base.digest(victim_path), "indices_sha256": base.digest(indices_path),
                "train_sha256": base.digest(raw / "cifar-100-python/train"),
                "transform": repr(transform), "temperature": 1.0, "classes": victim.last_linear.out_features}
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["metadata"] != metadata or not torch.equal(cached["indices"], indices):
            raise ValueError("已有查询缓存与 Victim、查询索引或输入处理不一致。")
    else:
        # 只取图片数组，原始训练标签不进入查询缓存和 B 的数据加载器。
        raw_images = datasets.CIFAR100(raw, train=True, download=False).data[indices.numpy()]
        images = torch.stack([transform(Image.fromarray(image)) for image in raw_images])
        victim.to(device).eval()
        with preserve_rng_state(), torch.inference_mode():
            probabilities = torch.cat([victim(images[start:start + batch_size].to(device)).softmax(dim=1).cpu()
                                       for start in range(0, len(images), batch_size)])
        victim.cpu()
        cached = {"metadata": metadata, "indices": indices, "images": images, "probabilities": probabilities}
        path.parent.mkdir(parents=True, exist_ok=True)
        base.save_checkpoint(path, cached)
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
    """在全部官方测试图片上评测，不消耗训练随机状态，不进行选模。"""
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
    blackbox = completed[0]["final_accuracy"]
    return [{**row, "gap_to_all_shield": round(row["final_accuracy"] - blackbox, 6)} for row in completed]


def summarize(completed, victim_accuracy, epsilon, inputs) -> dict:
    """完整扫描后分别确定两个策略及共同满足黑盒阈值的最小 k。"""
    blackbox = completed[0]["final_accuracy"]
    threshold = blackbox + epsilon
    by_strategy = {strategy: {row["k"]: row for row in completed if row["strategy"] == strategy}
                   for strategy in ("freeze", "joint")}
    critical = {}
    for strategy, rows in by_strategy.items():
        critical[strategy] = min((k for k, row in rows.items() if row["final_accuracy"] <= threshold), default=None)
    critical["both"] = min((k for k in by_strategy["freeze"]
                            if all(rows[k]["final_accuracy"] <= threshold for rows in by_strategy.values())), default=None)
    protection = {}
    for strategy, k in critical.items():
        if k is None:
            protection[strategy] = None
        else:
            row = by_strategy["freeze" if strategy == "both" else strategy][k]
            protection[strategy] = {key: row[key] for key in ("protected_parameters", "protected_ratio")}
    return {"victim_accuracy": victim_accuracy, "all_shield_accuracy": blackbox, "epsilon_pp": epsilon,
            "threshold_accuracy": threshold, "critical_k": critical, "critical_protection": protection,
            "inputs": inputs, "evaluation": "official_test_used_for_topk_selection"}


def run_scan(public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
             weights_dir, results_dir, *, resume=False) -> dict:
    """按完整训练组恢复；未完成的组从相同初值重训，避免改变随机序列。"""
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = weights_dir / "progress.pth"
    jobs = [("all_shield", None)] + [(strategy, k) for k in range(1, len(rank) + 1) for strategy in ("freeze", "joint")]
    progress = {"configuration": run_config, "completed": [], "history": []}
    if progress_path.exists():
        if not resume:
            raise FileExistsError("已有本次训练进度，请使用 --resume。")
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress["configuration"] != run_config:
            raise ValueError("恢复训练的输入、配置或代码与已保存进度不一致。")
    completed, history = progress["completed"], progress["history"]
    if [(row["strategy"], row["k"]) for row in completed] != jobs[:len(completed)]:
        raise ValueError("已完成训练组不符合规定的执行顺序。")
    if len(history) != len(completed) * training["epochs"]:
        raise ValueError("已完成训练组的逐轮记录不完整。")
    base.save_checkpoint(progress_path, progress)
    write_table(results_dir / "history.csv", history, HISTORY_FIELDS)
    device = torch.device(settings["device"])
    parameters = dict(public.named_parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters.values())
    dataset = TensorDataset(cached["images"], cached["probabilities"])
    for strategy, k in jobs[len(completed):]:
        seed_everything(b_config)
        protected = set(parameters) if strategy == "all_shield" else protected_names(public, rank, k)
        protected_count = sum(parameters[name].numel() for name in protected)
        model = make_student(public, victim, protected, strategy).to(device)
        loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                            num_workers=settings["num_workers"], worker_init_fn=seed_worker,
                            generator=torch_generator(b_config), pin_memory=device.type == "cuda",
                            drop_last=False, persistent_workers=False)
        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad],
                                    lr=training["lr"], momentum=training["momentum"], weight_decay=training["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["step_size"], gamma=training["gamma"])
        initial = evaluate(model, *evaluation, training["batch_size"])
        group = group_name(strategy, k)
        print(f"[{len(completed) + 1}/{len(jobs)}] {group}：初始准确率 {initial:.2f}%，"
              f"保护参数 {protected_count}/{total_parameters}。", flush=True)
        group_history = []
        for epoch in range(1, training["epochs"] + 1):
            started = time.monotonic()
            lr = optimizer.param_groups[0]["lr"]
            loss = train_epoch(model, loader, optimizer, device)
            accuracy = evaluate(model, *evaluation, training["batch_size"])
            scheduler.step()
            group_history.append({"strategy": strategy, "k": k, "epoch": epoch, "lr": lr,
                                  "train_loss": loss, "test_accuracy": accuracy, "seconds": time.monotonic() - started})
            if epoch == 1 or epoch % 10 == 0 or epoch == training["epochs"]:
                write_table(results_dir / "history.csv", history + group_history, HISTORY_FIELDS)
                print(f"  {group} {epoch:03d}/{training['epochs']}：loss={loss:.6f}，test={accuracy:.2f}%", flush=True)
        record = {"strategy": strategy, "k": k, "initial_accuracy": initial, "final_accuracy": accuracy,
                  "protected_parameters": protected_count, "protected_ratio": protected_count / total_parameters}
        base.save_checkpoint(weights_dir / f"{group}.pth", {
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "epoch": training["epochs"], "config": run_config, "protected": sorted(protected), **record})
        completed.append(record)
        history.extend(group_history)
        base.save_checkpoint(progress_path, progress)
        write_table(results_dir / "history.csv", history, HISTORY_FIELDS)
        write_table(results_dir / "accuracy.tsv", accuracy_rows(completed), ACCURACY_FIELDS, delimiter="\t")
        del model, optimizer, scheduler, loader
    summary = summarize(completed, run_config["victim_accuracy"], settings["epsilon_pp"], run_config["inputs"])
    keep = {"all_shield"}
    for strategy, k in summary["critical_k"].items():
        if k is not None:
            keep.update(group_name(item, k) for item in (("freeze", "joint") if strategy == "both" else (strategy,)))
    for strategy, k in jobs:
        name = group_name(strategy, k)
        path = weights_dir / f"{name}.pth"
        if name in keep:
            if not path.is_file():
                raise FileNotFoundError(f"已完成组缺少需保留的 Epoch 100 权重：{path}")
        else:
            path.unlink(missing_ok=True)
    summary["checkpoints"] = [str((weights_dir / f"{name}.pth").relative_to(ROOT)) for name in sorted(keep)]
    base.save_json(results_dir / "summary.json", summary)
    write_table(results_dir / "accuracy.tsv", accuracy_rows(completed), ACCURACY_FIELDS, delimiter="\t")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    parser.add_argument("--b-seed", type=int)
    parser.add_argument("--all-seeds", action="store_true", help="运行本实验声明的全部 B 重复种子")
    parser.add_argument("--resume", action="store_true", help="跳过已完成训练组，重新执行中断的组")
    parser.add_argument("--dry-run", action="store_true", help="只核对模型、Rank、查询索引和配置")
    args = parser.parse_args()
    if args.all_seeds and args.b_seed is not None:
        parser.error("--all-seeds 和 --b-seed 不能同时使用。")
    settings = read_settings(args.config)
    training_path = ROOT / settings["training_config"]
    training = base.load_training_config(training_path)
    a_config, query_config = load_config(seed=settings["a_seed"]), load_config(seed=settings["query_seed"])
    b_seeds = settings["b_seeds"] if args.all_seeds else [load_config(seed=args.b_seed).seed]
    seed_everything(a_config)
    torch.set_num_threads(settings["cpu_threads"])
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    public, victim, victim_path = load_models(settings, a_config)
    rank_path = ROOT / "results/02_rank" / settings["model"] / settings["dataset"] / f"seed_{a_config.seed}/rank.tsv"
    rank = load_rank(rank_path, public)
    indices_path = ROOT / "dataset/splits" / settings["dataset"] / f"seed_{query_config.seed}" / f"ratio_{query_config.ratio_name}/b_indices.txt"
    indices = read_indices(indices_path, query_config.b_ratio)
    relative = Path(settings["model"]) / settings["dataset"] / f"a_seed_{a_config.seed}"
    print(f"候选 {len(rank)} 项；每个 B seed 共 {2 * len(rank) + 1} 组，"
          f"每组 {training['epochs']} 轮；固定查询 {len(indices)} 张，无训练增强。", flush=True)
    if args.dry_run:
        return
    for seed in b_seeds:
        for parent in ("weights", "results"):
            path = ROOT / parent / "03_topk" / relative / f"b_seed_{seed}"
            if path.exists() and any(path.iterdir()) and not args.resume:
                raise FileExistsError(f"已有实验产物，请使用 --resume：{path}")
    public_path = fix_public_initialization(public, settings, a_config)
    query_path = ROOT / "dataset/queries" / relative / f"query_seed_{query_config.seed}" / f"ratio_{query_config.ratio_name}/probabilities.pth"
    cached = prepare_queries(query_path, indices, indices_path, victim, victim_path, settings, training["batch_size"], device)
    evaluation = load_evaluation(settings, device)
    victim_accuracy = evaluate(victim.to(device), *evaluation, training["batch_size"])
    victim.cpu()
    if not math.isclose(victim_accuracy, settings["victim"]["test_accuracy"], abs_tol=1e-9):
        raise ValueError("当前评测未复现 Victim 的准确率。")
    inputs = {name: identity(path) for name, path in {
        "rank": rank_path, "victim": victim_path, "public_init": public_path,
        "query_indices": indices_path, "query_cache": query_path}.items()}
    code = {name: base.digest(ROOT / name) for name in (
        "config.py", "randomness.py", "models/build.py", "models/weights.py",
        "exp/01_train_base/train.py", "exp/03_topk/train.py")}
    for seed in b_seeds:
        b_config = load_config(seed=seed)
        destination = relative / f"b_seed_{seed}"
        run_config = {"settings": settings, "training": {**training, "loss": "soft_cross_entropy"},
                      "a_seed": a_config.seed, "query_seed": query_config.seed, "b_seed": seed,
                      "b_ratio": query_config.ratio_name, "deterministic": b_config.deterministic,
                      "inputs": inputs, "training_config_sha256": base.digest(training_path),
                      "environment_sha256": base.digest(ROOT / "uv.lock"), "code_sha256": code,
                      "victim_accuracy": victim_accuracy, "train_transform": "cifar_test", "temperature": 1.0}
        weights_dir, results_dir = (ROOT / parent / "03_topk" / destination for parent in ("weights", "results"))
        summary = run_scan(public, victim, rank, cached, evaluation, settings, b_config, training, run_config,
                           weights_dir, results_dir, resume=args.resume)
        from plot import plot_results
        plot_results(results_dir)
        print(json.dumps({"b_seed": seed, "all_shield_accuracy": summary["all_shield_accuracy"],
                          "critical_k": summary["critical_k"]}, ensure_ascii=False), flush=True)
    for value in inputs.values():
        if base.digest(ROOT / value["path"]) != value["sha256"]:
            raise RuntimeError(f"实验输入文件被改写：{value['path']}")


if __name__ == "__main__":
    main()
