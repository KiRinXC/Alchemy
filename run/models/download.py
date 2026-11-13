#!/usr/bin/env python3
"""下载并校验四个固定版本的 ImageNet 预训练权重。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.weights import CHUNK_SIZE, WEIGHTS, WEIGHTS_ROOT, Weight, verify_weight


def download_weight(name: str, spec: Weight, destination: Path) -> None:
    """续传临时文件，校验通过后才将其保存为正式权重。"""
    if destination.exists() or destination.is_symlink():
        verify_weight(destination, spec)
        print(f"[{name}] 已有权重校验通过。", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"临时文件不能是符号链接：{partial}")
    for attempt in range(1, 4):
        if partial.exists() and partial.stat().st_size >= spec.size_bytes:
            try:
                verify_weight(partial, spec)
            except ValueError:
                partial.unlink()
            else:
                partial.replace(destination)
                print(f"[{name}] 已完成的临时权重校验通过。", flush=True)
                return
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Alchemy-model-downloader/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"[{name}] 下载 {attempt}/3，从 {offset / 2**20:.1f} MiB 开始。", flush=True)
        try:
            with urlopen(Request(spec.url, headers=headers), timeout=45) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{spec.size_bytes}"):
                        raise ValueError(f"续传范围不符：{content_range}")
                    mode = "ab"
                elif response.status == 200:
                    # 服务器忽略续传请求时重新写入，避免把完整文件追加到旧片段后。
                    mode = "wb"
                else:
                    raise ValueError(f"下载状态异常：{response.status}")
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        if time.monotonic() - last_report >= 15:
                            print(f"[{name}] {output.tell() / 2**20:.1f} / {spec.size_bytes / 2**20:.1f} MiB", flush=True)
                            last_report = time.monotonic()
            verify_weight(partial, spec)
            partial.replace(destination)
            print(f"[{name}] 下载完成，SHA-256 校验通过。", flush=True)
            return
        except (OSError, ValueError) as error:
            # 不完整的文件保留续传；完整但校验失败的文件已不能复用。
            if partial.exists() and partial.stat().st_size >= spec.size_bytes:
                try:
                    verify_weight(partial, spec)
                except ValueError:
                    partial.unlink()
            if attempt == 3:
                raise
            print(f"[{name}] {error}，稍后重试。", flush=True)
            time.sleep(2)


def write_manifest(root: Path) -> None:
    """记录当前已校验的本地权重及其固定来源。"""
    entries = []
    for name, spec in WEIGHTS.items():
        path = root / spec.filename
        if path.exists() or path.is_symlink():
            verify_weight(path, spec)
            entries.append({"model": name, "filename": spec.filename, "url": spec.url,
                            "bytes": spec.size_bytes, "sha256": spec.sha256, "classes": 1000})
    temporary = root / "manifest.json.tmp"
    try:
        temporary.write_text(json.dumps({"weights": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="模型名称；默认准备全部四种模型")
    parser.add_argument("--verify-only", action="store_true", help="只校验本地权重，不联网")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.models or WEIGHTS))
    unknown = set(names) - WEIGHTS.keys()
    if unknown:
        parser.error(f"未知模型：{', '.join(sorted(unknown))}；可选：{', '.join(WEIGHTS)}")
    if WEIGHTS_ROOT.is_symlink():
        raise ValueError(f"权重目录不能是符号链接：{WEIGHTS_ROOT}")
    for name in names:
        spec = WEIGHTS[name]
        path = WEIGHTS_ROOT / spec.filename
        if args.verify_only:
            verify_weight(path, spec)
            print(f"[{name}] 权重校验通过。", flush=True)
        else:
            download_weight(name, spec, path)
    if not args.verify_only:
        write_manifest(WEIGHTS_ROOT)


if __name__ == "__main__":
    main()
