#!/usr/bin/env python3
"""按 uv.lock 建立 Alchemy 的项目运行环境。"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
UV_VERSION = "0.13.7"
UV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/b9/93/"
    "0e053a2ca4523ee9825ccbedeb7121470d0a188933f11688b6d90d7f5c08/"
    "uv-0.13.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
# 官方发布的 wheel 文件 SHA-256：https://pypi.org/pypi/uv/0.13.7/json
UV_WHEEL_SHA256 = "2c9bdc4a872b60a93c5f26c0791089ea11585a4f9d3b3c017d78eb8487505e25"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_uv() -> Path:
    executable = ROOT / ".tools" / "uv"
    if executable.is_file():
        version = subprocess.check_output([str(executable), "--version"], text=True).split()[1]
        if version == UV_VERSION:
            return executable
        raise RuntimeError(f"Expected uv {UV_VERSION}, found {version} at {executable}")

    cache = ROOT / ".cache" / "bootstrap"
    cache.mkdir(parents=True, exist_ok=True)
    wheel = cache / Path(UV_WHEEL_URL).name
    if not wheel.exists():
        print(f"Downloading uv {UV_VERSION} into Alchemy ...", flush=True)
        partial = wheel.with_suffix(".part")
        with urlopen(UV_WHEEL_URL, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        partial.replace(wheel)
    if sha256(wheel) != UV_WHEEL_SHA256:
        raise RuntimeError(f"uv wheel checksum failed: {wheel}")

    executable.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        binaries = [name for name in archive.namelist() if name.endswith("/scripts/uv") or name == "uv/uv"]
        if len(binaries) != 1:
            raise RuntimeError("Cannot identify the uv binary in the verified wheel")
        temporary = executable.with_suffix(".tmp")
        with archive.open(binaries[0]) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        temporary.chmod(0o755)
        temporary.replace(executable)
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-lock", action="store_true", help="Explicitly resolve dependencies and update uv.lock")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("This environment targets WSL2/Linux x86_64.")
    if sys.version_info[:2] != (3, 10):
        parser.error("Run this script with Python 3.10.")
    if not args.update_lock and not (ROOT / "uv.lock").is_file():
        parser.error("uv.lock is missing; use --update-lock only when intentionally creating a new lock.")

    uv = str(ensure_uv())
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(ROOT / ".cache" / "uv")
    environment["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(*arguments: str) -> None:
        subprocess.run([uv, *arguments], cwd=ROOT, env=environment, check=True)

    if args.update_lock:
        run("lock", "--python", sys.executable)
    run("sync", "--locked", "--python", sys.executable)
    python = ROOT / ".venv" / "bin" / "python"
    run("pip", "check", "--python", str(python))
    subprocess.run([str(python), str(ROOT / "run/env/check.py")], cwd=ROOT, env=environment, check=True)
    subprocess.run([str(python), str(ROOT / "run/env/clean.py")], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
