#!/usr/bin/env python3
"""检查依赖库和 GPU 运算，在终端显示当前环境。"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
# 根据脚本位置导入项目根目录的公共工具。
sys.path.insert(0, str(ROOT))


def main() -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Run this check with Alchemy/.venv/bin/python.")
    os.environ["CUDA_CACHE_PATH"] = str(ROOT / ".cache" / "cuda")
    os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")

    import numpy as np
    import pretrainedmodels  # 检查旧版模型库能否正常导入。
    import torch
    from torchvision.ops import nms

    import config
    import randomness

    for module in (config, randomness):
        if Path(module.__file__).resolve() != ROOT / f"{module.__name__}.py":
            raise RuntimeError("导入的全局工具不属于当前项目。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the existing NVIDIA driver and WSL GPU access.")

    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    if not np.array_equal(torch.from_numpy(array).numpy(), array):
        raise RuntimeError("NumPy/PyTorch conversion failed.")
    values = torch.ones((16, 16), device="cuda", requires_grad=True)
    (values @ values).sum().backward()
    torch.cuda.synchronize()
    if values.grad is None or not torch.isfinite(values.grad).all().item():
        raise RuntimeError("CUDA backward check failed.")
    boxes = torch.tensor([[0, 0, 4, 4], [0, 0, 4, 4], [10, 10, 12, 12]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    if nms(boxes, scores, 0.5).tolist() != [0, 2]:
        raise RuntimeError("TorchVision CUDA operator check failed.")

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow", "pretrainedmodels", "setuptools", "six", "tomli", "tqdm")},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "checks": {"project_modules": "passed", "numpy_bridge": "passed", "cuda_backward": "passed", "torchvision_cuda_nms": "passed"},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("环境检查通过。")


if __name__ == "__main__":
    main()
