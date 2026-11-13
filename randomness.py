"""统一管理全局种子，为抽样和数据加载分别维护随机状态。"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import numpy as np

from config import ROOT, Config

if TYPE_CHECKING:
    import torch


def numpy_generator(config: Config) -> np.random.Generator:
    """创建独立的抽样生成器，避免消耗训练进程的随机状态。"""
    return np.random.Generator(np.random.PCG64(config.seed))


def seed_everything(config: Config) -> None:
    """每次独立实验开始时调用一次，在构造模型和初始化 CUDA 前设置随机性。"""
    import torch

    if config.deterministic:
        valid = {":4096:8", ":16:8"}
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in valid:
            if torch.cuda.is_initialized():
                raise RuntimeError("Set up reproducibility before initializing CUDA.")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_CACHE_PATH"] = str(ROOT / ".cache" / "cuda")
    os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  # 同时设置 CPU 和所有 CUDA 设备的随机种子。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.use_deterministic_algorithms(config.deterministic)


def torch_generator(config: Config) -> torch.Generator:
    """为一个 DataLoader 创建独立的 CPU 随机生成器，验证时另建生成器。"""
    import torch

    return torch.Generator().manual_seed(config.seed)


def seed_worker(worker_id: int) -> None:
    """沿用 DataLoader 派生的工作进程种子，同步设置 Python 和 NumPy 的随机性。"""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """保存并恢复训练进程的随机状态，避免验证或诊断改变后续随机序列。

    验证的数据加载器应使用独立的生成器。本函数不恢复外部 Generator 对象，
    也不切换 model.train()/eval() 模式。
    """
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
