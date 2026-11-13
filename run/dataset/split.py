#!/usr/bin/env python3
"""读取完整训练数据，抽取玩家 B 的样本并保存划分索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 根据脚本位置导入根目录的公共工具，支持从任意工作目录启动。
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import ROOT, Config, load_config
from randomness import numpy_generator


@dataclass(frozen=True)
class DatasetSpec:
    folder: str
    train_count: int
    class_count: int
    evaluation_split: str


DATASETS = {
    "c10": DatasetSpec("cifar-10-batches-py", 50_000, 10, "test"),
    "c100": DatasetSpec("cifar-100-python", 50_000, 100, "test"),
    "s10": DatasetSpec("stl10_binary", 5_000, 10, "test"),
    "t200": DatasetSpec("tiny-imagenet-200", 100_000, 200, "val"),
}


@dataclass(frozen=True)
class Sample:
    file: str  # 相对于 dataset/raw/<数据集>/ 的文件路径。
    row: int  # 批次或二进制文件中从 0 开始的样本行号；单张 JPEG 填 0。
    label: int  # 仅用于统计各类样本数量，不写入 B 的索引。


@dataclass(frozen=True)
class TrainingIndex:
    name: str
    samples: tuple[Sample, ...]
    classes: tuple[str, ...]
    raw_sha256: str
    archive_sha256: str


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_train_index(name: str, raw_root: Path = ROOT / "dataset/raw") -> TrainingIndex:
    """读取官方有标签训练数据，并在复用索引前计算输入文件的校验值。

    CIFAR 按官方批次和行号排列，STL 按二进制文件的行号排列，
    Tiny-ImageNet 按类别 ID 和类内文件名排序。
    """
    spec = DATASETS[name]
    root = raw_root / name
    provenance = json.loads((root / "source.json").read_text(encoding="utf-8"))
    archive_sha256 = provenance["archive_sha256"]
    if len(archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in archive_sha256):
        raise ValueError(f"{name}: invalid archive SHA-256 in source.json.")
    fingerprint = hashlib.sha256()

    def read(relative: str) -> bytes:
        content = (root / relative).read_bytes()
        fingerprint.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
        return content

    samples = []
    classes = tuple(str(label) for label in range(spec.class_count))
    if name in {"c10", "c100"}:
        filenames = [f"data_batch_{batch}" for batch in range(1, 6)] if name == "c10" else ["train"]
        label_key = "labels" if name == "c10" else "fine_labels"
        for filename in filenames:
            relative = f"{spec.folder}/{filename}"
            # 这里读取的是由 download.py 校验过的官方 pickle 文件。
            batch = pickle.loads(read(relative), encoding="latin1")
            labels = batch[label_key]
            if batch["data"].shape != (len(labels), 3072) or batch["data"].dtype != np.uint8:
                raise ValueError(f"{name}: unexpected image array in {relative}.")
            samples.extend(Sample(relative, row, int(label)) for row, label in enumerate(labels))
    elif name == "s10":
        relative = f"{spec.folder}/train_X.bin"
        images = read(relative)
        labels = np.frombuffer(read(f"{spec.folder}/train_y.bin"), dtype=np.uint8)
        if len(images) != len(labels) * 3 * 96 * 96:
            raise ValueError("s10: image/label counts disagree.")
        samples.extend(Sample(relative, row, int(label) - 1) for row, label in enumerate(labels))
    else:
        classes = tuple(sorted(read(f"{spec.folder}/wnids.txt").decode("utf-8").splitlines()))
        if len(classes) != spec.class_count or len(set(classes)) != spec.class_count:
            raise ValueError("t200: unexpected class IDs in wnids.txt.")
        train = root / spec.folder / "train"
        if {path.name for path in train.iterdir() if path.is_dir()} != set(classes):
            raise ValueError("t200: training directories do not match wnids.txt.")
        for label, class_id in enumerate(classes):
            images = sorted((train / class_id / "images").glob("*.JPEG"))
            if len(images) != spec.train_count // spec.class_count:
                raise ValueError(f"t200: unexpected image count for {class_id}.")
            for image in images:
                relative = image.relative_to(root).as_posix()
                read(relative)
                samples.append(Sample(relative, 0, label))

    if len(samples) != spec.train_count:
        raise ValueError(f"{name}: expected {spec.train_count} labeled training samples, found {len(samples)}.")
    counts = Counter(sample.label for sample in samples)
    expected = {label: spec.train_count // spec.class_count for label in range(spec.class_count)}
    if counts != expected:
        raise ValueError(f"{name}: official training labels have unexpected class counts: {dict(counts)}.")
    return TrainingIndex(name, tuple(samples), classes, fingerprint.hexdigest(), archive_sha256)


def select_indices(size: int, config: Config) -> tuple[np.ndarray, np.ndarray]:
    numerator, denominator = config.b_ratio.as_integer_ratio()
    count = size * numerator // denominator
    if count < 1:
        raise ValueError(f"b_ratio={config.ratio_name} selects zero samples from {size}; increase it.")
    order = numpy_generator(config).permutation(size)
    return order, order[:count]


def write_immutable(path: Path, content: bytes) -> None:
    """完整写入文件；已有结果或并发生成的结果不一致时拒绝覆盖。"""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"Concurrent result differs; refusing to overwrite: {path}")
        finally:
            temporary.unlink(missing_ok=True)


def write_split(
    index: TrainingIndex, config: Config, output_root: Path = ROOT / "dataset/splits"
) -> dict:
    """保存样本索引、数据来源、实际配置和校验记录。"""
    order, selected = select_indices(len(index.samples), config)
    train_content = ("index\tfile\trow\n" + "".join(
        f"{number}\t{sample.file}\t{sample.row}\n" for number, sample in enumerate(index.samples)
    )).encode("utf-8")
    order_content = "".join(f"{int(number)}\n" for number in order).encode("ascii")
    selected_content = "".join(f"{int(number)}\n" for number in selected).encode("ascii")
    source = {
        "schema_version": 1,
        "dataset": index.name,
        "pool": "official_labeled_train",
        "a_train_count": len(index.samples),
        "classes_in_label_order": list(index.classes),
        "evaluation_split": DATASETS[index.name].evaluation_split,
        "archive_sha256": index.archive_sha256,
        "training_inputs_sha256": index.raw_sha256,
        "train_index_sha256": sha256(train_content),
        "sample_paths_relative_to": f"dataset/raw/{index.name}",
    }
    class_counts = Counter(index.samples[int(number)].label for number in selected)
    implementation = hashlib.sha256()
    for relative in ("config.py", "randomness.py", "run/dataset/split.py"):
        implementation.update(relative.encode("utf-8") + b"\0" + (ROOT / relative).read_bytes())
    manifest = {
        "schema_version": 1,
        "dataset": index.name,
        "seed": config.seed,
        "b_ratio": config.ratio_name,
        "a_train_count": len(index.samples),
        "b_train_count": len(selected),
        "b_is_subset_of_a": True,
        "a_keeps_all_training_samples": True,
        "b_labels": "query_A_full_probabilities_not_generated_yet",
        "evaluation_split": DATASETS[index.name].evaluation_split,
        "sampling": "numpy.PCG64.permutation; prefix floor(N * ratio); without replacement; not stratified",
        "numpy_version": np.__version__,
        "effective_config": config.snapshot(),
        "b_class_counts_for_audit_only": {label: class_counts[number] for number, label in enumerate(index.classes)},
        "hashes": {
            "train_source_sha256": sha256(json_bytes(source)),
            "train_index_sha256": sha256(train_content),
            "order_sha256": sha256(order_content),
            "b_indices_sha256": sha256(selected_content),
            "implementation_sha256": implementation.hexdigest(),
            "uv_lock_sha256": sha256((ROOT / "uv.lock").read_bytes()),
        },
    }
    directory = output_root / index.name
    seed_directory = directory / f"seed_{config.seed}"
    ratio_directory = seed_directory / f"ratio_{config.ratio_name}"
    artifacts = {
        directory / "train_index.tsv": train_content,
        directory / "train_source.json": json_bytes(source),
        seed_directory / "order.txt": order_content,
        ratio_directory / "b_indices.txt": selected_content,
        ratio_directory / "manifest.json": json_bytes(manifest),
    }
    # 先核对全部已有文件，再写入缺少的文件。
    for path, content in artifacts.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing split or provenance differs; refusing to overwrite: {path}")
    for path, content in artifacts.items():
        write_immutable(path, content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="c10 c100 s10 t200; omitted or all selects all four")
    parser.add_argument("--config", type=Path, help="Alternate TOML file; defaults to Alchemy/config.toml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seed", type=int, help="Override the global seed for this run")
    mode.add_argument("--all-seeds", action="store_true", help="Generate each seed listed in config.toml")
    parser.add_argument("--b-ratio", help="Fraction of A's training data, e.g. 0.01 or 0.05")
    args = parser.parse_args()
    names = list(DATASETS) if not args.datasets or args.datasets == ["all"] else args.datasets
    if any(name not in DATASETS for name in names) or len(set(names)) != len(names):
        parser.error("Choose distinct datasets from c10, c100, s10, t200, or use all alone.")
    try:
        config = load_config(args.config, seed=args.seed, b_ratio=args.b_ratio)
        seeds = config.seeds if args.all_seeds else (config.seed,)
        for name in names:
            print(f"[{name}] Checking official training inputs and building the stable index ...", flush=True)
            index = build_train_index(name)
            for seed in seeds:
                manifest = write_split(index, config.for_seed(seed))
                print(
                    f"[{name}] seed={seed}, ratio={config.ratio_name}: "
                    f"A={manifest['a_train_count']}, B={manifest['b_train_count']}; "
                    f"dataset/splits/{name}/seed_{seed}/ratio_{config.ratio_name}/",
                    flush=True,
                )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Split failed: {error}\n")


if __name__ == "__main__":
    main()
