"""保留 TensorShield 原始逐批更新、归一化与排序顺序，计算全部 weight 的重要性。"""

from __future__ import annotations

import copy

import torch
import torch.optim as optim

SOURCE_URL = "https://github.com/suntong30/TensorShield/blob/main/model-stealing/knockoff/adversary/get_importance.py"
SOURCE_BLOB = "15f764470e304c5bf17385161a7aaaf7d4d10a77"


def compute_intrinsic(victim, train_loader, *, epochs=10, lr=0.01, momentum=0.9, weight_decay=0.0):
    """返回作者顺序的全部 weight 分数及更新次数；输入 Victim 保持不变。"""
    if weight_decay != 0:
        raise ValueError("作者原始 SGD 使用 weight_decay=0。")
    pretrained = copy.deepcopy(victim)
    model = copy.deepcopy(pretrained)
    # 输入已加载原始 Victim；等价于作者 load_target_model 的权重载入及 eval。
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    device = next(model.parameters()).device
    loss_function = torch.nn.CrossEntropyLoss()

    # 两个完整缓冲区及默认参数精度均沿用作者实现，包括未参与评分的 bias 槽位。
    weight_updates = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    weight_importance = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        current_update = (-optimizer.param_groups[0]['lr'] * param.grad).abs()
                        param_grad_abs = param.grad.abs()
                        weight_updates[name].add_(current_update)
                        weight_importance[name].addcmul_(current_update, param_grad_abs)
                        weight_importance[name].div_(param.numel())
        sorted_importances = {k: v.sum().item() for k, v in weight_importance.items()}
        sorted_importances = sorted(sorted_importances.items(), key=lambda item: item[1], reverse=True)
        print(f"原始内部重要性 {epoch + 1}/{epochs}：累计更新 {(epoch + 1) * len(train_loader)} 次", flush=True)

    sorted_importances = {k: v.sum().item() for k, v in weight_importance.items()}
    sorted_importances = sorted(sorted_importances.items(), key=lambda item: item[1], reverse=True)
    # 保留作者中间筛选与排序；最终原始 weight 排名仍取自完整排序。
    filtered_importances = {k: v.sum().item() for k, v in weight_importance.items() if 'bn' not in k and 'downsample' not in k}
    sorted_filtered_importances = sorted(filtered_importances.items(), key=lambda item: item[1], reverse=True)
    new_select = [k for k in sorted_importances if "weight" in k[0]]
    return dict(new_select), epochs * len(train_loader)
