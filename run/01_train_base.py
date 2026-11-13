#!/usr/bin/env python3
"""复现基础微调流程：校验输入、准备 A/B 索引、全量训练并用测试集选模。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tomli

from config import load_config
from models.weights import WEIGHTS


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-B", str(ROOT / script), *arguments], cwd=ROOT, check=True)


def main() -> None:
    with (ROOT / "exp/01_train_base/config.toml").open("rb") as stream:
        settings = tomli.load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WEIGHTS, default=settings["model"])
    parser.add_argument("--dataset", choices=("c10", "c100", "s10", "t200"), default=settings["dataset"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seed", type=int)
    mode.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--training-config", type=Path, default=ROOT / settings["training_config"])
    parser.add_argument("--device", default=settings["device"])
    parser.add_argument("--num-workers", type=int, default=settings["num_workers"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(seed=args.seed)
    run("run/dataset/download.py", args.dataset, "--verify-only")
    run("run/models/download.py", args.model, "--verify-only")
    for seed in config.seeds if args.all_seeds else (config.seed,):
        run("run/dataset/split.py", args.dataset, "--seed", str(seed))
        options = ["--model", args.model, "--dataset", args.dataset, "--seed", str(seed),
                   "--training-config", str(args.training_config.resolve()), "--device", args.device,
                   "--num-workers", str(args.num_workers)]
        if args.resume:
            options.append("--resume")
        if args.dry_run:
            options.append("--dry-run")
        run("exp/01_train_base/train.py", *options)


if __name__ == "__main__":
    main()
