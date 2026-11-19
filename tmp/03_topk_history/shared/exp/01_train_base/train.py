#!/usr/bin/env python3
"""使用全部官方训练数据微调玩家 A，并按官方测试集准确率选模。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import tomli
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import Config, load_config
from models.build import MODELS, NUM_CLASSES, build_model
from models.weights import WEIGHTS
from randomness import preserve_rng_state, seed_everything, seed_worker, torch_generator

EXPERIMENT = "01_train_base"
SETTINGS_PATH = Path(__file__).with_name("config.toml")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_config(path: Path) -> dict:
    """只从共享配置读取训练超参数，拒绝拼错或缺失的字段。"""
    with path.open("rb") as stream:
        config = tomli.load(stream)
    expected = {"epochs", "batch_size", "loss", "optimizer", "lr", "momentum", "weight_decay", "scheduler", "step_size", "gamma"}
    if set(config) != expected:
        raise ValueError(f"训练配置必须包含且仅包含：{', '.join(sorted(expected))}")
    for key in ("epochs", "batch_size", "step_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} 必须是正整数。")
    for key in ("lr", "momentum", "weight_decay", "gamma"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} 必须是有限数值。")
    if not (config["lr"] > 0 and 0 <= config["momentum"] < 1 and config["weight_decay"] >= 0 and 0 < config["gamma"] <= 1):
        raise ValueError("学习率、动量、权重衰减或衰减系数超出允许范围。")
    if (config["loss"], config["optimizer"], config["scheduler"]) != ("cross_entropy", "sgd", "step"):
        raise ValueError("当前微调配置使用 cross_entropy、sgd 和 step。")
    return config


def build_transforms(name: str) -> tuple:
    """按 TensorShield 公开 MS 代码的数据集映射构造变换。"""
    # 作者将 Tiny-ImageNet 映射到 CIFAR：训练裁剪为 32，评估保留原始尺寸。
    if name in {"c10", "c100", "t200"}:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        evaluation = []
    elif name == "s10":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train = [transforms.RandomResizedCrop(96), transforms.RandomHorizontalFlip()]
        evaluation = [transforms.Resize(128), transforms.CenterCrop(96)]
    else:
        raise ValueError(f"未知数据集：{name}")
    tail = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train + tail), transforms.Compose(evaluation + tail)


class TinyValidation(Dataset):
    """读取 Tiny-ImageNet 官方验证图片与标签，用于测试选模。"""

    def __init__(self, root: Path, class_to_idx: dict, transform) -> None:
        self.transform = transform
        self.samples = []
        for line in (root / "val_annotations.txt").read_text().splitlines():
            filename, class_id, *_ = line.split()
            self.samples.append((root / "images" / filename, class_to_idx[class_id]))
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, number: int) -> tuple:
        path, label = self.samples[number]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def build_datasets(name: str, model_name: str) -> tuple:
    """保持官方训练文件的样本顺序，分别为完整训练集和测试集设置变换。"""
    train_transform, evaluation_transform = build_transforms(name)
    # 沿用作者训练入口：VGG 在测试选模时也使用训练增强。
    if "vgg" in model_name or "alex" in model_name:
        evaluation_transform = train_transform
    root = ROOT / "dataset/raw" / name
    if name in {"c10", "c100"}:
        factory = datasets.CIFAR10 if name == "c10" else datasets.CIFAR100
        train = factory(root, train=True, transform=train_transform, download=False)
        test = factory(root, train=False, transform=evaluation_transform, download=False)
    elif name == "s10":
        train = datasets.STL10(root, split="train", transform=train_transform, download=False)
        test = datasets.STL10(root, split="test", transform=evaluation_transform, download=False)
    else:
        root = root / "tiny-imagenet-200"
        train = datasets.ImageFolder(root / "train", transform=train_transform)
        test = TinyValidation(root / "val", train.class_to_idx, evaluation_transform)
    if len(train.classes) != NUM_CLASSES[name]:
        raise ValueError("数据集类别数与模型分类头不符。")
    return train, test


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, config: Config, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=False, persistent_workers=False,
                      worker_init_fn=seed_worker, generator=torch_generator(config))


def train_epoch(model, loader, optimizer, device: torch.device) -> dict:
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += logits.argmax(1).eq(labels).sum().item()
        count += len(labels)
    return {"loss": loss_sum / count, "accuracy": 100 * correct / count}


def evaluate(model, loader, device: torch.device) -> dict:
    """评估不消耗训练进程的全局随机状态，并恢复原有训练模式。"""
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    try:
        with preserve_rng_state(), torch.inference_mode():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(inputs)
                loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                correct += logits.argmax(1).eq(labels).sum().item()
                count += len(labels)
    finally:
        model.train(was_training)
    return {"loss": loss_sum / count, "accuracy": 100 * correct / count, "count": count}


def capture_random_state(train_loader, test_loader) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            "train_loader": train_loader.generator.get_state(), "test_loader": test_loader.generator.get_state()}


def restore_random_state(state: dict, train_loader, test_loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_loader"])
    # 兼容迁入前断点使用的评估生成器字段。
    test_loader.generator.set_state(state["test_loader"] if "test_loader" in state else state["val_loader"])


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_history(path: Path, history: list) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(weights_dir: Path, run_config: dict) -> dict:
    """从权重文件核对实际训练配置，结果目录不再承担断点校验。"""
    last = torch.load(weights_dir / "last.pth", map_location="cpu", weights_only=False)
    best = torch.load(weights_dir / "best.pth", map_location="cpu", weights_only=True)
    if "config" in last:
        saved_config = last["config"]
        if best.get("config") != saved_config:
            raise ValueError("最佳模型与最近断点不属于同一组配置。")
    else:
        # 兼容已经完成的旧格式权重，读取原有配置，不重写模型文件。
        original = best["metadata"]
        if last.get("signature") != best.get("signature") or original.get("selection_rule") != "highest_accuracy_then_latest_epoch":
            raise ValueError("旧断点与当前选模协议不一致。")
        keys = ("model", "dataset", "seed", "training", "num_workers", "cpu_threads", "device",
                "training_inputs_sha256", "pretrained_sha256")
        saved_config = {key: original[key] for key in keys}
        saved_config.update({"deterministic": original["global_config"]["random"]["deterministic"],
                             "train_count": original["counts"]["train"],
                             "test_count": original["counts"]["selection_test"],
                             "environment_sha256": original["code_sha256"]["uv.lock"]})
    if saved_config != run_config:
        raise ValueError("断点与当前模型、数据或训练设置不一致。")
    if best["epoch"] != last["best"]["epoch"] or best["selection"] != last["best"]:
        raise ValueError("最佳模型与最近断点记录的选模结果不一致。")
    if [row["epoch"] for row in last["history"]] != list(range(1, last["epoch"] + 1)):
        raise ValueError("断点中的逐轮指标不完整。")
    return last


def fit(model, train_loader, test_loader, training: dict, device: torch.device,
        weights_dir: Path, results_dir: Path, run_config: dict, *, resume: bool = False, stop_after: int | None = None) -> list:
    """按测试准确率选模，同分保留较晚轮次，返回完整逐轮指标。"""
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last_path = weights_dir / "last.pth"
    occupied = any(weights_dir.glob("*.pth")) or any(results_dir.iterdir())
    if occupied and not resume:
        raise FileExistsError("已有训练产物，请使用 --resume。")
    if occupied and not last_path.exists():
        raise FileNotFoundError("已有训练产物但缺少 last.pth，拒绝覆盖。")
    optimizer = torch.optim.SGD(model.parameters(), lr=training["lr"], momentum=training["momentum"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["step_size"], gamma=training["gamma"])
    history, best, start_epoch = [], {"epoch": 0, "accuracy": -1.0, "loss": None}, 1
    if resume and last_path.exists():
        checkpoint = load_training_state(weights_dir, run_config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_random_state(checkpoint["random_state"], train_loader, test_loader)
        history, best, start_epoch = checkpoint["history"], checkpoint["best"], checkpoint["epoch"] + 1
        save_history(results_dir / "history.csv", history)
    last_epoch = training["epochs"] if stop_after is None else min(stop_after, training["epochs"])
    for epoch in range(start_epoch, last_epoch + 1):
        started = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        if test_metrics["accuracy"] >= best["accuracy"]:
            best = {"epoch": epoch, "accuracy": test_metrics["accuracy"], "loss": test_metrics["loss"]}
            save_checkpoint(weights_dir / "best.pth", {"state_dict": model.state_dict(), "epoch": epoch,
                            "selection_split": "official_val" if run_config.get("dataset") == "t200" else "official_test",
                            "selection": best, "config": run_config})
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"], "test_loss": test_metrics["loss"],
                        "test_accuracy": test_metrics["accuracy"], "seconds": time.monotonic() - started})
        save_checkpoint(last_path, {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                        "best": best, "random_state": capture_random_state(train_loader, test_loader), "config": run_config})
        save_history(results_dir / "history.csv", history)
        print(f"[{run_config.get('model', 'model')}/{run_config.get('dataset', 'data')}] "
              f"{epoch:03d}/{training['epochs']} lr={lr:g} train={train_metrics['accuracy']:.2f}% "
              f"test={test_metrics['accuracy']:.2f}% best={best['accuracy']:.2f}% "
              f"time={history[-1]['seconds']:.1f}s", flush=True)
    return history


def summarize(history: list) -> dict:
    """提取最佳轮和最后一轮的指标，耗时为各轮训练与评估的累计值。"""
    if not history:
        raise ValueError("没有逐轮指标，无法汇总结果。")

    def metrics(row: dict) -> dict:
        return {"epoch": int(row["epoch"]),
                "train_accuracy": round(float(row["train_accuracy"]), 3),
                "test_accuracy": round(float(row["test_accuracy"]), 3),
                "train_loss": round(float(row["train_loss"]), 6),
                "test_loss": round(float(row["test_loss"]), 6)}

    best = max(history, key=lambda row: (float(row["test_accuracy"]), int(row["epoch"])))
    return {"best": metrics(best), "last": metrics(history[-1]),
            "train_eval_minutes": round(sum(float(row["seconds"]) for row in history) / 60, 3)}


def build_run_config(args, settings: dict, config: Config, training: dict, train_count: int, test_count: int) -> dict:
    """实际训练参数只写入权重文件，用于恢复时检查配置是否一致。"""
    source = json.loads((ROOT / "dataset/splits" / args.dataset / "train_source.json").read_text())
    if source["a_train_count"] != train_count:
        raise ValueError("完整训练集的样本数与索引记录不一致。")
    return {"model": args.model, "dataset": args.dataset, "seed": config.seed,
            "deterministic": config.deterministic, "training": training,
            "num_workers": args.num_workers, "cpu_threads": settings["cpu_threads"], "device": str(torch.device(args.device)),
            "train_count": train_count, "test_count": test_count,
            "training_inputs_sha256": source["training_inputs_sha256"],
            "pretrained_sha256": WEIGHTS[args.model].sha256, "environment_sha256": digest(ROOT / "uv.lock")}


def main() -> None:
    with SETTINGS_PATH.open("rb") as stream:
        settings = tomli.load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=settings["model"])
    parser.add_argument("--dataset", choices=NUM_CLASSES, default=settings["dataset"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--training-config", type=Path, default=ROOT / settings["training_config"])
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--num-workers", type=int, default=settings["num_workers"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="核对模型、数据和配置，不执行训练")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers 不能为负数。")
    config = load_config(seed=args.seed)
    training = load_training_config(args.training_config)
    seed_everything(config)
    torch.set_num_threads(settings["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    train_data, test_data = build_datasets(args.dataset, args.model)
    train_loader = make_loader(train_data, training["batch_size"], True, args.num_workers, config, device)
    test_loader = make_loader(test_data, training["batch_size"], False, args.num_workers, config, device)
    model = build_model(args.model, args.dataset, config).to(device)
    relative = Path(EXPERIMENT) / args.model / args.dataset / f"seed_{config.seed}"
    weights_dir, results_dir = ROOT / "weights" / relative, ROOT / "results" / relative
    run_config = build_run_config(args, settings, config, training, len(train_data), len(test_data))
    print(f"{args.model}/{args.dataset}/seed_{config.seed}：训练 {len(train_data)} 张，测试选模 {len(test_data)} 张。", flush=True)
    if args.dry_run:
        return
    history = fit(model, train_loader, test_loader, training, device, weights_dir, results_dir, run_config, resume=args.resume)
    result = summarize(history)
    save_json(results_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
