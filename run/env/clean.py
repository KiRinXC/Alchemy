#!/usr/bin/env python3
"""验证结束后清理临时测试文件、安装缓存和源码字节码。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def remove_directory(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        print(f"Removed {path.relative_to(ROOT)}/", flush=True)


def main() -> None:
    # 测试目录只保留说明，脚本、数据和结果均为一次性验证产物。
    tests = ROOT / "tests"
    if tests.is_dir():
        for path in tests.iterdir():
            if path.name == "README.md":
                continue
            if path.is_dir() and not path.is_symlink():
                remove_directory(path)
            else:
                path.unlink()
                print(f"Removed {path.relative_to(ROOT)}", flush=True)
    uv = ROOT / ".tools/uv"
    cache = ROOT / ".cache/uv"
    if uv.is_file():
        # 通过 uv 自身命令清理，遵守缓存锁，避免误删正在使用的条目。
        subprocess.run([str(uv), "cache", "clean", "--cache-dir", str(cache), "--offline"], cwd=ROOT, check=True)
        remove_directory(ROOT / ".cache/bootstrap")
    remove_directory(ROOT / ".cache/matplotlib")

    # 清理旧打包方式遗留的项目元数据和源码字节码。
    remove_directory(ROOT / "alchemy_research.egg-info")
    remove_directory(ROOT / "__pycache__")
    for name in ("run", "models", "tests", "exp", "lab"):
        for path in (ROOT / name).rglob("__pycache__"):
            remove_directory(path)
    print("Kept .venv, .tools, raw data/archives, model caches and experiment outputs.")


if __name__ == "__main__":
    main()
