"""在 Victim 副本上累计实际 SGD 更新的贡献，计算内部重要性。"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


def compute_intrinsic(
    victim: nn.Module,
    loader: DataLoader,
    candidates: list[str],
    *,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> tuple[dict[str, float], int]:
    """返回候选张量的 G_k 和实际更新次数，原始 Victim 保持不变。"""
    model = copy.deepcopy(victim).eval()
    device = next(model.parameters()).device
    all_parameters = dict(model.named_parameters())
    parameters = {name: all_parameters[name] for name in candidates}
    totals = {name: torch.zeros((), dtype=torch.float64, device=device) for name in candidates}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    steps = 0
    for epoch in range(1, epochs + 1):
        loss_sum, count = 0.0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("内部重要性计算出现非有限损失。")
            loss.backward()
            before = {name: param.detach().clone() for name, param in parameters.items()}
            optimizer.step()
            with torch.no_grad():
                for name, param in parameters.items():
                    if param.grad is None:
                        raise RuntimeError(f"候选张量缺少梯度：{name}")
                    delta = param - before[name]
                    contribution = (param.grad * delta).abs().sum(dtype=torch.float64)
                    totals[name].add_(contribution)
            steps += 1
            count += len(labels)
            loss_sum += loss.item() * len(labels)
        print(f"内部重要性 {epoch}/{epochs}：loss={loss_sum / count:.6f}，累计更新 {steps} 次", flush=True)
    if steps == 0:
        raise ValueError("内部重要性需要至少一个有效批次。")
    # T 表示全部轮次的实际优化器更新次数，最后统一除以 T 和张量参数数。
    scores = {name: total.item() / (steps * parameters[name].numel()) for name, total in totals.items()}
    if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in scores.values()):
        raise RuntimeError("内部重要性包含非有限分数。")
    return scores, steps
