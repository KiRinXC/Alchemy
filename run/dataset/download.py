#!/usr/bin/env python3
"""下载并校验 Alchemy 的四个原始数据集，仅使用 Python 标准库。

在 Alchemy 中运行：python3 run/dataset/download.py
指定数据集：python3 run/dataset/download.py c10 c100
离线校验：python3 run/dataset/download.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
ARCHIVE_ROOT = PROJECT_ROOT / "dataset" / ".archives"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Dataset:
    title: str
    url: str
    filename: str
    folder: str
    md5: str
    checksum_source: str
    size_bytes: int
    ignored_files: tuple[str, ...] = ()


DATASETS = {
    "c10": Dataset(
        "CIFAR-10",
        "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1",
        "cifar-10-python.tar.gz",
        "cifar-10-batches-py",
        "c58f30108f718f92721af3b95e74349a",
        "https://www.cs.toronto.edu/~kriz/cifar.html",
        170498071,
    ),
    "c100": Dataset(
        "CIFAR-100",
        "https://zenodo.org/records/10089977/files/cifar-100-python.tar.gz?download=1",
        "cifar-100-python.tar.gz",
        "cifar-100-python",
        "eb9058c3a382ffc7106e4002c42a8d85",
        "https://www.cs.toronto.edu/~kriz/cifar.html",
        169001437,
        ("cifar-100-python/file.txt~",),  # 官方压缩包中附带的空编辑器备份文件。
    ),
    "s10": Dataset(
        "STL-10",
        "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        "stl10_binary.tar.gz",
        "stl10_binary",
        "91f7769df0f17e558f3565bffb0c7dfb",
        "https://github.com/pytorch/vision/blob/v0.16.1/torchvision/datasets/stl10.py",
        2640397119,
    ),
    "t200": Dataset(
        "Tiny-ImageNet-200",
        "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "tiny-imagenet-200.zip",
        "tiny-imagenet-200",
        "90528d7ca1a48142e341f4ef8d21d0de",
        "https://zenodo.org/records/10720917",
        248100043,
    ),
}


def log(name: str, message: str) -> None:
    print(f"[{name}] {message}", flush=True)


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, spec: Dataset, archive: Path) -> None:
    """续传未完成的下载，通过 MD5 校验后再保存为正式压缩包。"""
    if archive.exists():
        if checksum(archive) != spec.md5:
            raise ValueError(f"Archive checksum failed: {archive}. Remove it and retry.")
        log(name, "Archive checksum OK; no download needed.")
        return

    partial = archive.with_name(archive.name + ".part")
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Alchemy-dataset-downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        log(name, f"Download attempt {attempt}/4, resume at {offset / 2**20:.1f} MiB.")
        try:
            with urlopen(Request(spec.url, headers=headers), timeout=45) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif response.status == 200:
                    # 部分服务器忽略 Range 请求，此时重新写入，避免重复追加内容。
                    offset, mode = 0, "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                length = response.headers.get("Content-Length")
                expected_size = offset + int(length) if length else None
                received, last_report = offset, time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report >= 20:
                            total = f" / {expected_size / 2**20:.1f} MiB" if expected_size else ""
                            log(name, f"Downloaded {received / 2**20:.1f} MiB{total}")
                            last_report = time.monotonic()
                if expected_size is not None and received != expected_size:
                    raise OSError(f"Incomplete download: {received} / {expected_size} bytes")
            if partial.stat().st_size != spec.size_bytes or checksum(partial) != spec.md5:
                partial.unlink()
                raise OSError("Downloaded archive checksum failed; retrying from the start.")
            partial.replace(archive)
            log(name, "Download complete; archive checksum OK.")
            return
        except HTTPError as error:
            # 下载可能已完成，但在重命名前中断，留下完整的 .part 文件。
            if error.code == 416 and partial.exists() and checksum(partial) == spec.md5:
                partial.replace(archive)
                return
            if attempt == 4:
                raise
            log(name, f"HTTP error: {error}; retrying.")
        except (OSError, URLError) as error:
            if attempt == 4:
                raise
            log(name, f"Download interrupted: {error}; retrying.")
        time.sleep(3)


def process_archive(archive: Path, destination: Path, spec: Dataset, *, verify: bool) -> dict:
    """解压普通文件，或逐字节核对解压文件与原始压缩包。"""
    is_zip = archive.suffix == ".zip"
    opener = zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r|gz")
    file_count, total_bytes = 0, 0
    with opener as bundle:
        members = bundle.infolist() if is_zip else bundle
        for member in members:
            member_name = member.filename if is_zip else member.name
            relative = PurePosixPath(member_name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != spec.folder:
                raise ValueError(f"Unexpected archive path: {member_name}")
            if member_name in spec.ignored_files:
                size = member.file_size if is_zip else member.size
                if size != 0:
                    raise ValueError(f"Expected an empty archive backup: {member_name}")
                continue
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError(f"Archive path escapes destination: {member_name}")
            is_directory = member.is_dir() if is_zip else member.isdir()
            if is_directory:
                if not verify:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            if is_zip:
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Archive contains a symlink: {member_name}")
                size = member.file_size
                source = bundle.open(member)
            else:
                if not member.isfile():
                    raise ValueError(f"Archive contains a non-regular file: {member_name}")
                size = member.size
                source = bundle.extractfile(member)
            with source:
                if verify:
                    if not target.is_file() or target.stat().st_size != size:
                        raise ValueError(f"Missing file or incorrect size: {target}")
                    with target.open("rb") as actual:
                        while expected := source.read(CHUNK_SIZE):
                            if actual.read(len(expected)) != expected:
                                raise ValueError(f"File content differs from archive: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output, CHUNK_SIZE)
            file_count += 1
            total_bytes += size
    return {"file_count": file_count, "extracted_bytes": total_bytes}


def prepare(name: str, verify_only: bool) -> None:
    spec = DATASETS[name]
    archive = ARCHIVE_ROOT / spec.filename
    destination = RAW_ROOT / name
    if archive.is_symlink() or destination.is_symlink():
        raise ValueError("Use independent local files and directories, not symlinks to another project.")
    if verify_only:
        if not archive.is_file() or not destination.is_dir():
            raise FileNotFoundError(f"{name} is not downloaded. Run without --verify-only first.")
        if checksum(archive) != spec.md5:
            raise ValueError(f"Archive checksum failed: {archive}")
    else:
        download(name, spec, archive)
        if not destination.exists():
            log(name, "Extracting the original archive ...")
            with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=RAW_ROOT) as temporary:
                staging = Path(temporary) / name
                staging.mkdir()
                process_archive(archive, staging, spec, verify=False)
                staging.rename(destination)
        for member_name in spec.ignored_files:
            backup = destination / member_name
            if backup.is_symlink():
                raise ValueError(f"Refusing to remove a symlink: {backup}")
            if backup.exists():
                if not backup.is_file() or backup.stat().st_size != 0:
                    raise ValueError(f"Refusing to remove a non-empty backup: {backup}")
                backup.unlink()
                log(name, f"Removed empty archive backup: {member_name}")

    log(name, "Verifying every extracted file against the original archive ...")
    summary = process_archive(archive, destination, spec, verify=True)
    if not verify_only:
        manifest = {
            "dataset": spec.title,
            "download_url": spec.url,
            "archive": str(archive.relative_to(PROJECT_ROOT)),
            "archive_bytes": archive.stat().st_size,
            "archive_md5": spec.md5,
            "archive_sha256": checksum(archive, "sha256"),
            "checksum_source": spec.checksum_source,
            "ignored_archive_files": list(spec.ignored_files),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        temporary_manifest = destination / ".source.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(destination / "source.json")
    log(name, f"OK: {summary['file_count']} files, {summary['extracted_bytes']:,} original bytes.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help="c10 c100 s10 t200; default: all")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing data offline without writing files")
    args = parser.parse_args()
    names = list(DATASETS) if not args.datasets or args.datasets == ["all"] else list(dict.fromkeys(args.datasets))
    unknown = set(names) - DATASETS.keys()
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(sorted(unknown))}")
    if not args.verify_only:
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        tasks = {executor.submit(prepare, name, args.verify_only): name for name in names}
        for task in as_completed(tasks):
            name = tasks[task]
            try:
                task.result()
            except Exception as error:
                failures.append(name)
                log(name, f"FAILED: {error}")
    if failures:
        print(f"Failed datasets: {', '.join(sorted(failures))}", file=sys.stderr)
        return 1
    print(f"All requested datasets are ready in {RAW_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
