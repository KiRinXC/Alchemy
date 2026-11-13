"""四个 ImageNet 预训练权重的固定版本、存放位置与完整性校验。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = ROOT / "weights" / "pre_train"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Weight:
    filename: str
    size_bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://download.pytorch.org/models/{self.filename}"


# 版本与 TensorShield 的 pretrainedmodels 0.7.4 及 MobileNetV2 加载代码一致。
# 完整校验值来自已核对的 TSDP 清单，并在下载后重新计算核验。
WEIGHTS = {
    "resnet18": Weight(
        "resnet18-5c106cde.pth", 46827520,
        "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
    ),
    "resnet50": Weight(
        "resnet50-19c8e357.pth", 102502400,
        "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097",
    ),
    "vgg16_bn": Weight(
        "vgg16_bn-6c64b313.pth", 553507836,
        "6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a",
    ),
    "mobilenetv2": Weight(
        "mobilenet_v2-b0353104.pth", 14212972,
        "b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889",
    ),
}


def verify_weight(path: Path, spec: Weight) -> None:
    """检查独立权重文件的字节数和完整 SHA-256。"""
    if path.is_symlink():
        raise ValueError(f"权重必须是项目内的独立文件：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"缺少权重 {path}，请先运行 run/models/download.py。")
    if path.stat().st_size != spec.size_bytes:
        raise ValueError(f"权重大小不符：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ValueError(f"权重 SHA-256 不符：{path}")
