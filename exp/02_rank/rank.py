"""读取固定数据排列与玩家 A，计算两项评分并保存综合排名。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import tomli
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import Config, load_config
from models.build import build_model
from models.weights import WEIGHTS
from randomness import seed_everything, seed_worker, torch_generator

from attention import compute_attention
from intrinsic import compute_intrinsic

# 来源：TSDP 中作者确认的候选 rank；仅复制名称，不在运行时依赖 TSDP。
AUTHOR_RANK = (
    "layer1.1.conv1.weight", "layer2.0.conv1.weight", "last_linear.weight",
    "layer1.0.conv1.weight", "layer1.1.conv2.weight", "layer2.0.conv2.weight",
    "layer2.1.conv1.weight", "layer1.0.conv2.weight", "layer3.0.conv1.weight",
    "layer2.1.conv2.weight", "layer3.0.conv2.weight", "layer4.0.conv1.weight",
    "layer4.0.conv2.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight",
    "layer3.1.conv2.weight", "layer3.1.conv1.weight",
)
OUTPUTS = ("intrinsic.tsv", "attention.tsv", "rank.tsv")


def read_settings(path: Path) -> dict:
    with path.open("rb") as stream:
        settings = tomli.load(stream)
    if (settings["model"], settings["dataset"]) != ("resnet18", "c100"):
        raise ValueError("当前协议与作者对照限定为 resnet18+c100。")
    integers = [settings["cpu_threads"], settings["intrinsic"]["epochs"],
                settings["intrinsic"]["batch_size"], settings["attention"]["batch_size"]]
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("线程数、轮数和批次大小必须是正整数。")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("num_workers 必须是非负整数。")
    training = settings["intrinsic"]
    if not all(math.isfinite(training[key]) for key in ("lr", "momentum", "weight_decay")):
        raise ValueError("优化器参数必须有限。")
    if not (training["lr"] > 0 and 0 <= training["momentum"] < 1 and training["weight_decay"] >= 0):
        raise ValueError("优化器参数超出有效范围。")
    if not 0 < settings["data"]["intrinsic_fraction"] < 1:
        raise ValueError("数据比例必须在 0 和 1 之间。")
    return settings


def load_models(settings: dict, config: Config) -> tuple[nn.Module, nn.Module]:
    """按 A 的同一训练起点构造 M_0，再将已训练权重加载到独立的 M_A。"""
    checkpoint_path = ROOT / "weights/01_train_base" / settings["model"] / settings["dataset"] / f"seed_{config.seed}/best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("config", checkpoint.get("metadata", {}))
    for key, expected in (("model", settings["model"]), ("dataset", settings["dataset"]), ("seed", config.seed)):
        if metadata.get(key) != expected:
            raise ValueError(f"A 权重的 {key} 与本次配置不符。")
    selection = checkpoint["selection"]
    if checkpoint["epoch"] != settings["victim"]["epoch"] or not math.isclose(
        selection["accuracy"], settings["victim"]["test_accuracy"], abs_tol=1e-9
    ):
        raise ValueError("A 权重的轮次或选模准确率与协议不符。")
    if metadata["pretrained_sha256"] != WEIGHTS[settings["model"]].sha256:
        raise ValueError("公共预训练权重与 A 的训练起点不符。")
    # 旧权重记录了构建代码指纹，核对构建逻辑，环境可增加本实验所需依赖。
    for relative in ("config.py", "randomness.py", "models/build.py", "models/weights.py"):
        expected = metadata.get("code_sha256", {}).get(relative)
        if expected is not None and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"模型初始化代码与 A 保存的记录不同：{relative}")
    public = build_model(settings["model"], settings["dataset"], config)
    victim = build_model(settings["model"], settings["dataset"], config)
    victim.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(settings["device"])
    return public.to(device).eval(), victim.to(device).eval()


def candidate_names(model: nn.Module) -> list[str]:
    names = [f"{name}.weight" for name, module in model.named_modules()
             if isinstance(module, (nn.Conv2d, nn.Linear)) and name != "conv1" and "downsample" not in name]
    if set(names) != set(AUTHOR_RANK):
        raise ValueError("候选集合与协议规定的十七项不符。")
    return names


def load_data(settings: dict, config: Config) -> tuple[DataLoader, DataLoader]:
    """直接读取现有 order.txt，并复用 01_train_base 的训练与评估变换。"""
    base = importlib.import_module("exp.01_train_base.train")
    train_transform, test_transform = base.build_transforms(settings["dataset"])
    raw = ROOT / "dataset/raw" / settings["dataset"]
    train = datasets.CIFAR100(raw, train=True, transform=train_transform, download=False)
    attention = datasets.CIFAR100(raw, train=True, transform=test_transform, download=False)
    order_path = ROOT / "dataset/splits" / settings["dataset"] / f"seed_{config.seed}/order.txt"
    order = np.loadtxt(order_path, dtype=np.int64, ndmin=1)
    if not np.array_equal(np.sort(order), np.arange(len(train))):
        raise ValueError("order.txt 必须恰好包含完整训练池的无重复排列。")
    cut = math.floor(len(order) * settings["data"]["intrinsic_fraction"])
    attention_counts = np.bincount(np.asarray(attention.targets)[order[cut:]], minlength=100)
    if np.any(attention_counts == 0):
        raise ValueError("固定注意力子集缺少类别，无法按一百类等权平均。")
    print(f"固定划分：内部重要性 {cut} 张，注意力 {len(order)-cut} 张；两个子集无重叠。", flush=True)
    print(f"注意力子集每类 {attention_counts.min()}–{attention_counts.max()} 张；输入 3×32×32。", flush=True)

    def loader(dataset, indices, batch_size, shuffle):
        return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=shuffle,
                          num_workers=settings["num_workers"], worker_init_fn=seed_worker,
                          generator=torch_generator(config), pin_memory=settings["device"].startswith("cuda"),
                          drop_last=False, persistent_workers=False)

    return (loader(train, order[:cut], settings["intrinsic"]["batch_size"], True),
            loader(attention, order[cut:], settings["attention"]["batch_size"], False))


def snapshot(model: nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def check_unchanged(model: nn.Module, original: dict, name: str) -> None:
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), original[key]):
            raise RuntimeError(f"{name} 的参数或缓冲区发生变化：{key}")


def combine_scores(intrinsic: dict, attention: dict) -> list[dict]:
    """按综合评分降序排列全部候选，并附上作者排名。"""
    rows = [{"tensor": name, "G_k": intrinsic[name], "A_k": attention[name]["A_k"],
             "I_k": intrinsic[name] * attention[name]["A_k"]} for name in intrinsic]
    if not all(math.isfinite(row["I_k"]) and row["I_k"] >= 0 for row in rows):
        raise ValueError("综合评分必须是有限非负数。")
    rows.sort(key=lambda row: (-row["I_k"], row["tensor"]))
    for rank, row in enumerate(rows, 1):
        row.update(rank=rank, author_rank=AUTHOR_RANK.index(row["tensor"]) + 1)
    return rows


def write_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="只核对模型、固定划分和候选集合")
    parser.add_argument("--overwrite", action="store_true", help="重新计算并替换已有评分结果")
    args = parser.parse_args()
    settings, config = read_settings(args.config), load_config(seed=args.seed)
    seed_everything(config)
    torch.set_num_threads(settings["cpu_threads"])
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    output = ROOT / "results/02_rank" / settings["model"] / settings["dataset"] / f"seed_{config.seed}"
    if not args.dry_run and not args.overwrite and any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("评分结果已存在；需要重新计算时显式使用 --overwrite。")
    intrinsic_loader, attention_loader = load_data(settings, config)
    public, victim = load_models(settings, config)
    candidates = candidate_names(victim)
    print(f"A：epoch {settings['victim']['epoch']}，准确率 {settings['victim']['test_accuracy']}%；候选 {len(candidates)} 项。", flush=True)
    if args.dry_run:
        return
    public_before, victim_before = snapshot(public), snapshot(victim)
    options = {key: value for key, value in settings["intrinsic"].items() if key != "batch_size"}
    intrinsic, steps = compute_intrinsic(victim, intrinsic_loader, candidates, **options)
    check_unchanged(public, public_before, "M_0")
    check_unchanged(victim, victim_before, "M_A")
    attention = compute_attention(public, victim, attention_loader, candidates, 100)
    check_unchanged(public, public_before, "M_0")
    check_unchanged(victim, victim_before, "M_A")
    rows = combine_scores(intrinsic, attention)
    parameters = dict(victim.named_parameters())
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "intrinsic.tsv", [{"tensor": name, "parameters": parameters[name].numel(), "G_k": intrinsic[name]}
                                          for name in candidates], ["tensor", "parameters", "G_k"])
    write_table(output / "attention.tsv", [{"tensor": name, **attention[name]} for name in candidates],
                ["tensor", "A_k", "kind", "both_zero", "one_zero"])
    write_table(output / "rank.tsv", rows, ["rank", "tensor", "G_k", "A_k", "I_k", "author_rank"])
    print(f"评分完成：内部重要性 {steps} 次更新；M_0、M_A 的全部参数和缓冲区保持一致。", flush=True)
    print(f"已保存 {len(rows)} 个候选的评分和完整排名：{output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
