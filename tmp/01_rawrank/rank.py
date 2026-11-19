"""隔离运行原始内部重要性与完整注意力排序，全部产物写入当前实验目录。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
# 第三方库的缓存和进程间临时文件也必须留在本实验目录内。
for variable, folder in {"MPLCONFIGDIR": ".cache/matplotlib", "TORCH_HOME": ".cache/torch",
                         "CUDA_CACHE_PATH": ".cache/cuda", "XDG_CACHE_HOME": ".cache",
                         "TMPDIR": ".cache/tmp"}.items():
    os.environ[variable] = str(HERE / folder)
(HERE / ".cache/tmp").mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(HERE / ".cache/tmp")

import numpy as np
import tomli
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import randomness
from config import load_config
from models.build import build_model
from models.weights import WEIGHTS

from attention import compute_attention
from intrinsic import SOURCE_BLOB, SOURCE_URL, compute_intrinsic

# 仅在本进程重定向共享随机性工具的缓存位置，保留其全部种子管理逻辑。
randomness.ROOT = HERE

# 来源：TSDP/exp/MS/train_surrogate/selector/tensorshield.py 中作者确认的完整排序。
AUTHOR_RANK = (
    "layer1.0.bn2.weight", "bn1.weight", "layer1.1.bn2.weight", "layer2.0.bn2.weight",
    "layer1.1.bn1.weight", "layer2.1.bn1.weight", "layer2.1.bn2.weight", "layer1.0.bn1.weight",
    "layer2.0.downsample.1.weight", "layer2.0.bn1.weight", "layer3.0.bn1.weight", "layer3.0.bn2.weight",
    "layer3.0.downsample.1.weight", "conv1.weight", "layer2.0.downsample.0.weight", "layer4.0.bn1.weight",
    "layer1.1.conv1.weight", "layer2.0.conv1.weight", "last_linear.weight", "layer1.0.conv1.weight",
    "layer1.1.conv2.weight", "layer4.0.downsample.1.weight", "layer2.0.conv2.weight", "layer2.1.conv1.weight",
    "layer1.0.conv2.weight", "layer3.0.conv1.weight", "layer2.1.conv2.weight", "layer4.0.bn2.weight",
    "layer4.1.bn1.weight", "layer3.0.downsample.0.weight", "layer4.0.downsample.0.weight", "layer3.0.conv2.weight",
    "layer3.1.bn2.weight", "layer4.1.bn2.weight", "layer4.0.conv1.weight", "layer3.1.bn1.weight",
    "layer4.0.conv2.weight", "layer4.1.conv1.weight", "layer4.1.conv2.weight", "layer3.1.conv2.weight",
    "layer3.1.conv1.weight",
)
# 来源：TensorShield 官方 README 的 ResNet18+CIFAR-100 保护集合。
AUTHOR_TOP10 = (
    "layer1.0.conv1.weight", "layer1.0.conv2.weight", "layer1.1.conv1.weight", "layer1.1.conv2.weight",
    "layer2.0.conv1.weight", "layer2.0.conv2.weight", "layer2.1.conv1.weight", "layer2.1.conv2.weight",
    "layer3.0.conv1.weight", "last_linear.weight",
)
OUTPUTS = ("intrinsic.tsv", "attention.tsv", "rank.tsv", "rank_filtered.tsv", "summary.json")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_text(path, text):
    """以临时文件替换结果，拒绝写入实验目录以外的位置。"""
    if not path.resolve().is_relative_to(HERE):
        raise ValueError(f"输出必须位于 {HERE} 内：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, stream.getvalue())


def read_settings(path):
    with path.open("rb") as stream:
        settings = tomli.load(stream)
    if (settings["model"], settings["dataset"]) != ("resnet18", "c100"):
        raise ValueError("本次协议只支持 resnet18+c100。")
    expected = {"epochs": 10, "batch_size": 128, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0}
    if settings["intrinsic"] != expected:
        raise ValueError("内部重要性必须保留作者的 10 轮、128 批次和原始 SGD 设置。")
    if settings["top_k"] != 10 or settings["data"]["intrinsic_fraction"] != 0.9:
        raise ValueError("本次协议要求固定 90%/10% 划分和 Top-10。")
    for value in (settings["cpu_threads"], settings["attention"]["batch_size"]):
        if type(value) is not int or value <= 0:
            raise ValueError("线程数和注意力批次大小必须为正整数。")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("数据加载进程数必须为非负整数。")
    return settings


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def check_unchanged(model, before, name):
    if any(not torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items()):
        raise RuntimeError(f"{name} 的参数或缓冲区发生变化。")


def load_models(settings, config):
    """固定公共初始化并加载既有 Victim，所有新权重只写入本目录。"""
    relative = Path(settings["model"]) / settings["dataset"] / f"seed_{config.seed}"
    victim_path = ROOT / "weights/exp/01_train_base" / relative / "best.pth"
    checkpoint = torch.load(victim_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("config", checkpoint.get("metadata", {}))
    for key, expected in (("model", settings["model"]), ("dataset", settings["dataset"]), ("seed", config.seed)):
        if metadata.get(key) != expected:
            raise ValueError(f"Victim 的 {key} 与本次配置不符。")
    if checkpoint["epoch"] != settings["victim"]["epoch"] or not math.isclose(
        checkpoint["selection"]["accuracy"], settings["victim"]["test_accuracy"], abs_tol=1e-9
    ):
        raise ValueError("Victim 轮次或选模准确率与配置不符。")
    if metadata["pretrained_sha256"] != WEIGHTS[settings["model"]].sha256:
        raise ValueError("Victim 与公共预训练权重不一致。")
    for name in ("config.py", "randomness.py", "models/build.py", "models/weights.py"):
        expected = metadata.get("code_sha256", {}).get(name)
        if expected is not None and digest(ROOT / name) != expected:
            raise ValueError(f"模型初始化代码与 Victim 记录不一致：{name}")

    public = build_model(settings["model"], settings["dataset"], config)
    initial = snapshot(public)
    external = ROOT / "weights/public_init" / relative / "init.pth"
    initial_path = external if external.is_file() else HERE / "weights/public_init" / relative / "init.pth"
    if initial_path.is_file():
        saved = torch.load(initial_path, map_location="cpu", weights_only=True)
        public.load_state_dict(saved.get("state_dict", saved), strict=True)
        check_unchanged(public, initial, "公共初始化与同种子构建结果")
    else:
        if not initial_path.resolve().is_relative_to(HERE):
            raise ValueError("新初始化文件必须位于当前实验目录。")
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = initial_path.with_suffix(".pth.tmp")
        try:
            torch.save({"state_dict": initial, "model": settings["model"], "dataset": settings["dataset"],
                        "seed": config.seed, "pretrained_sha256": WEIGHTS[settings["model"]].sha256}, temporary)
            temporary.replace(initial_path)
        finally:
            temporary.unlink(missing_ok=True)
    victim = build_model(settings["model"], settings["dataset"], config)
    victim.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(settings["device"])
    return public.to(device).eval(), victim.to(device).eval(), initial_path, victim_path


def load_data(settings, config):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR100(ROOT / "dataset/raw/c100", train=True, transform=train_transform, download=False)
    attention = datasets.CIFAR100(ROOT / "dataset/raw/c100", train=True, transform=test_transform, download=False)
    order_path = ROOT / "dataset/splits/c100" / f"seed_{config.seed}/order.txt"
    order = np.loadtxt(order_path, dtype=np.int64, ndmin=1)
    if not np.array_equal(np.sort(order), np.arange(len(train))):
        raise ValueError("固定排列必须包含完整训练集且无重复。")
    cut = math.floor(len(order) * settings["data"]["intrinsic_fraction"])
    counts = np.bincount(np.asarray(attention.targets)[order[cut:]], minlength=100)
    if np.any(counts <= 0):
        raise ValueError("注意力子集必须覆盖全部 100 类。")

    def loader(dataset, indices, batch_size, shuffle):
        return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=shuffle,
                          num_workers=settings["num_workers"], worker_init_fn=randomness.seed_worker,
                          generator=randomness.torch_generator(config), drop_last=False, persistent_workers=False)

    print(f"固定划分：内部重要性 {cut} 张，注意力 {len(order) - cut} 张；每类 {counts.min()}–{counts.max()} 张。", flush=True)
    return (loader(train, order[:cut], settings["intrinsic"]["batch_size"], True),
            loader(attention, order[cut:], settings["attention"]["batch_size"], False), order_path, counts)


def eligible(model, name):
    module = model.get_submodule(name.removesuffix(".weight"))
    return not isinstance(module, nn.modules.batchnorm._BatchNorm) and "downsample" not in name and name != "conv1.weight"


def combine_scores(model, intrinsic, attention):
    rows = [{"tensor": name, "G_k": value, "A_k": attention[name]["A_k"], "I_k": value * attention[name]["A_k"],
             "eligible": eligible(model, name), "author_rank": AUTHOR_RANK.index(name) + 1}
            for name, value in intrinsic.items()]
    if any(not math.isfinite(row["I_k"]) or row["I_k"] < 0 for row in rows):
        raise ValueError("综合分数必须是有限非负值。")
    rows.sort(key=lambda row: row["I_k"], reverse=True)
    eligible_rank = 0
    for position, row in enumerate(rows, 1):
        if row["eligible"]:
            eligible_rank += 1
        row.update(rank=position, eligible_rank=eligible_rank if row["eligible"] else "")
    return rows


def filter_rank(rows):
    """保留可选候选，将本次排名和作者排名分别重新编号。"""
    candidates = [row for row in rows if row["eligible"]]
    author_positions = {row["tensor"]: position for position, row in
                        enumerate(sorted(candidates, key=lambda row: row["author_rank"]), 1)}
    return [{"rank": position, "tensor": row["tensor"], "G_k": row["G_k"], "A_k": row["A_k"],
             "I_k": row["I_k"], "author_rank": author_positions[row["tensor"]]}
            for position, row in enumerate(candidates, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="准备固定初始化并核对输入，不执行评分")
    parser.add_argument("--overwrite", action="store_true", help="重新计算并替换本实验结果")
    args = parser.parse_args()
    os.chdir(HERE)
    settings, config = read_settings(HERE / "config.toml"), load_config(seed=args.seed)
    randomness.seed_everything(config)
    torch.set_num_threads(settings["cpu_threads"])
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA 设备。")
    output = HERE / "results" / settings["model"] / settings["dataset"] / f"seed_{config.seed}"
    if not args.dry_run and not args.overwrite and any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("结果已存在；重新计算请使用 --overwrite。")
    intrinsic_loader, attention_loader, order_path, class_counts = load_data(settings, config)
    public, victim, initial_path, victim_path = load_models(settings, config)
    weights = {name: param for name, param in victim.named_parameters() if "weight" in name}
    if set(weights) != set(AUTHOR_RANK) or sum(eligible(victim, name) for name in weights) != 17:
        raise ValueError("模型必须包含作者对应的 41 个 weight 和 17 个可选候选。")
    author_eligible = [name for name in AUTHOR_RANK if eligible(victim, name)]
    if set(author_eligible[:10]) != set(AUTHOR_TOP10):
        raise ValueError("作者完整 Rank 与公布的 Top-10 集合不一致。")
    print(f"Victim epoch {settings['victim']['epoch']}，准确率 {settings['victim']['test_accuracy']}%；共 {len(weights)} 个 weight。", flush=True)
    if args.dry_run:
        return

    inputs = {str(path.relative_to(ROOT)): digest(path) for path in (initial_path, victim_path, order_path)}
    public_before, victim_before = snapshot(public), snapshot(victim)
    started = time.monotonic()
    options = {key: value for key, value in settings["intrinsic"].items() if key != "batch_size"}
    intrinsic, steps = compute_intrinsic(victim, intrinsic_loader, **options)
    intrinsic_seconds = time.monotonic() - started
    check_unchanged(public, public_before, "M_0")
    check_unchanged(victim, victim_before, "M_A")
    started = time.monotonic()
    attention = compute_attention(public, victim, attention_loader, list(weights), 100)
    attention_seconds = time.monotonic() - started
    check_unchanged(public, public_before, "M_0")
    check_unchanged(victim, victim_before, "M_A")
    if any(digest(ROOT / name) != value for name, value in inputs.items()):
        raise RuntimeError("输入 checkpoint 或固定划分被改写。")
    rows = combine_scores(victim, intrinsic, attention)
    selected = [row["tensor"] for row in rows if row["eligible"]][:settings["top_k"]]
    protected = [*selected, "last_linear.bias"] if "last_linear.weight" in selected else selected.copy()
    common = [name for name in selected if name in AUTHOR_TOP10]
    comparison = {"intersection_count": len(common), "intersection": common,
                  "computed_only": [name for name in selected if name not in AUTHOR_TOP10],
                  "author_only": [name for name in AUTHOR_TOP10 if name not in selected],
                  "top10_equal": set(selected) == set(AUTHOR_TOP10),
                  "rank_equal": [row["tensor"] for row in rows] == list(AUTHOR_RANK),
                  "eligible_rank_equal": [row["tensor"] for row in rows if row["eligible"]] == author_eligible}
    summary = {"configuration": {**settings, "seed": config.seed, "deterministic": config.deterministic},
               "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__, "grad_cam": version("grad-cam")},
               "inputs_sha256": inputs, "intrinsic_source": {"url": SOURCE_URL, "git_blob": SOURCE_BLOB},
               "counts": {"weights": len(weights), "eligible": len(author_eligible), "intrinsic_images": len(intrinsic_loader.dataset),
                          "attention_images": len(attention_loader.dataset), "intrinsic_steps": steps,
                          "attention_class_min": int(class_counts.min()), "attention_class_max": int(class_counts.max())},
               "top10": selected, "protected_tensors": protected, "author_comparison": comparison,
               "checks": {"models_unchanged": True, "input_files_unchanged": True},
               "seconds": {"intrinsic": round(intrinsic_seconds, 3), "attention": round(attention_seconds, 3)}}
    write_table(output / "intrinsic.tsv", [{"rank": i, "tensor": name, "parameters": weights[name].numel(), "G_k": value}
                                         for i, (name, value) in enumerate(intrinsic.items(), 1)], ["rank", "tensor", "parameters", "G_k"])
    write_table(output / "attention.tsv", [{"tensor": name, **attention[name]} for name in weights],
                ["tensor", "A_k", "kind", "both_zero", "one_zero"])
    write_table(output / "rank.tsv", rows, ["rank", "tensor", "G_k", "A_k", "I_k", "eligible", "eligible_rank", "author_rank"])
    write_table(output / "rank_filtered.tsv", filter_rank(rows), ["rank", "tensor", "G_k", "A_k", "I_k", "author_rank"])
    write_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"完成：{steps} 次原始 SGD 更新，41 项完整 Rank；Top-10 与作者交集 {len(common)}/10。", flush=True)
    print("Top-10：" + ", ".join(selected), flush=True)
    print(f"结果：{output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
