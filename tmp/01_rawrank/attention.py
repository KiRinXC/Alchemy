"""对全部卷积、BN 和分类头 weight 计算注意力转移。"""

from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torch.utils.data import DataLoader


class LayerCapture:
    """按模块保存原始输出和梯度，避免残差分支的回调顺序混淆。"""

    def __init__(self, model, layers):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.handles = [layer.register_forward_hook(self.capture) for layer in layers]

    def capture(self, module, inputs, output):
        # BN 输出随后可能被原地 ReLU 或残差加法修改，立即保留原始值。
        self.activations[module] = output.detach().cpu().clone()
        if output.requires_grad:
            def save_gradient(gradient):
                self.gradients[module] = gradient.detach().cpu().clone()
            output.register_hook(save_gradient)

    def __call__(self, images):
        self.activations.clear()
        self.gradients.clear()
        return self.model(images)

    def release(self):
        for handle in self.handles:
            handle.remove()


class RawGradCAM(GradCAM):
    """复用库的前向、梯度和通道权重计算，返回 ReLU 后未归一化的响应。"""

    def __init__(self, model, target_layers):
        super().__init__(model, target_layers)
        self.activations_and_grads.release()
        self.activations_and_grads = LayerCapture(model, target_layers)

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth=False):
        if eigen_smooth:
            raise ValueError("本协议不使用特征平滑。")
        activations = self.activations_and_grads.activations
        gradients = self.activations_and_grads.gradients
        if len(activations) != len(self.target_layers) or len(gradients) != len(self.target_layers):
            raise RuntimeError("候选模块的特征与梯度数量不一致。")
        responses = []
        self.response_kinds = []
        for layer in self.target_layers:
            feature, gradient = activations[layer].numpy(), gradients[layer].numpy()
            if feature.ndim != 4 or feature.shape != gradient.shape:
                raise ValueError("卷积和 BN 模块必须产生四维特征及对应梯度。")
            if feature.shape[-2:] == (1, 1):
                # 保留每个通道的 alpha_c * feature_c，再逐通道 ReLU，避免压成单个标量。
                weights = self.get_cam_weights(input_tensor, layer, targets, feature, gradient)
                response = weights * feature[:, :, 0, 0]
                kind = "channel"
            elif min(feature.shape[-2:]) >= 2:
                response = self.get_cam_image(input_tensor, layer, targets, feature, gradient)
                kind = "spatial"
            else:
                raise ValueError(f"协议未定义这种空间尺寸：{feature.shape[-2:]}")
            responses.append(np.maximum(response, 0))
            self.response_kinds.append(kind)
        return responses

    def aggregate_multi_layers(self, cam_per_target_layer):
        # 各层保持自己的空间分辨率或通道维度，不做缩放、min-max 或跨层平均。
        return cam_per_target_layer


@torch.no_grad()
def head_contribution(head: nn.Linear, features: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """返回真实类别的 ReLU(h_i × W_yi)，即分类头输入的正向通道贡献。"""
    if features.ndim != 2 or features.shape[1] != head.in_features or labels.shape != features.shape[:1]:
        raise ValueError("分类头输入或目标类别的形状不匹配。")
    weights = head.weight[labels.to(head.weight.device)]
    return (features * weights).relu().cpu().numpy()


def cosine_distance(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐图计算余弦差异；双零响应记零，单零响应记一。"""
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("对应层的两份注意力响应形状必须相同。")
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("注意力响应包含非有限数值。")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    both_zero = (left_norm == 0) & (right_norm == 0)
    one_zero = (left_norm == 0) ^ (right_norm == 0)
    valid = (left_norm > 0) & (right_norm > 0)
    cosine = np.zeros(len(left), dtype=np.float64)
    np.divide((left * right).sum(axis=1), left_norm * right_norm, out=cosine, where=valid)
    distance = 1 - np.clip(cosine, 0, 1)
    distance[both_zero] = 0
    distance[one_zero] = 1
    return distance, both_zero, one_zero


def class_balanced_mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """先在类别内部平均，再对各类别等权平均。"""
    if np.any(counts <= 0):
        raise ValueError("注意力子集必须包含全部类别。")
    return (sums / counts[:, None]).mean(axis=0)


def compute_attention(
    public: nn.Module,
    victim: nn.Module,
    loader: DataLoader,
    candidates: list[str],
    num_classes: int,
) -> dict[str, dict]:
    """只进行前向和反向传播，所有样本都计入对应类别的统计。"""
    public.eval()
    victim.eval()
    device = next(victim.parameters()).device
    feature_names = [name for name in candidates if name != "last_linear.weight"]
    names = [*feature_names, "last_linear.weight"]
    public_layers = [public.get_submodule(name.removesuffix(".weight")) for name in feature_names]
    victim_layers = [victim.get_submodule(name.removesuffix(".weight")) for name in feature_names]
    if not isinstance(public.last_linear, nn.Linear) or not isinstance(victim.last_linear, nn.Linear):
        raise ValueError("本协议的分类头必须是线性层。")

    head_inputs = {}

    def save_head_input(module, inputs):
        # 复用 Grad-CAM 前向时的分类头输入，保留两模型各自的特征。
        head_inputs[module] = inputs[0].detach()

    sums = np.zeros((num_classes, len(names)), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    both_zeros = np.zeros(len(names), dtype=np.int64)
    one_zeros = np.zeros(len(names), dtype=np.int64)
    seen = 0
    kinds = None
    last_print = time.monotonic()
    with public.last_linear.register_forward_pre_hook(save_head_input), \
         victim.last_linear.register_forward_pre_hook(save_head_input), \
         RawGradCAM(model=public, target_layers=public_layers) as public_cam, \
         RawGradCAM(model=victim, target_layers=victim_layers) as victim_cam:
        for images, labels in loader:
            head_inputs.clear()
            images = images.to(device, non_blocking=True)
            classes = labels.numpy()
            if np.any(classes < 0) or np.any(classes >= num_classes):
                raise ValueError("注意力目标类别超出模型输出范围。")
            targets = [ClassifierOutputTarget(int(label)) for label in classes]
            left = public_cam(input_tensor=images, targets=targets)
            right = victim_cam(input_tensor=images, targets=targets)
            if public_cam.response_kinds != victim_cam.response_kinds:
                raise ValueError("两个模型的对应模块使用了不同类型的响应。")
            left.append(head_contribution(public.last_linear, head_inputs[public.last_linear], labels))
            right.append(head_contribution(victim.last_linear, head_inputs[victim.last_linear], labels))
            batch_kinds = [*public_cam.response_kinds, "linear"]
            if kinds is not None and kinds != batch_kinds:
                raise ValueError("不同批次的注意力响应类型不一致。")
            kinds = batch_kinds
            distances = []
            for i, (public_response, victim_response) in enumerate(zip(left, right)):
                distance, both_zero, one_zero = cosine_distance(public_response, victim_response)
                distances.append(distance)
                both_zeros[i] += both_zero.sum()
                one_zeros[i] += one_zero.sum()
            np.add.at(sums, classes, np.stack(distances, axis=1))
            counts += np.bincount(classes, minlength=num_classes)
            seen += len(labels)
            if time.monotonic() - last_print >= 15 or seen == len(loader.dataset):
                print(f"注意力迁移：{seen}/{len(loader.dataset)}", flush=True)
                last_print = time.monotonic()
    means = class_balanced_mean(sums, counts)
    result = {name: {"A_k": float(means[i]), "kind": kinds[i],
                     "both_zero": int(both_zeros[i]), "one_zero": int(one_zeros[i])}
              for i, name in enumerate(names)}
    public.zero_grad(set_to_none=True)
    victim.zero_grad(set_to_none=True)
    return {name: result[name] for name in candidates}
