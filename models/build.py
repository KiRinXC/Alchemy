"""加载预训练主体，并用全局种子初始化任务分类头。

结构和参数命名参照 TSDP/models/imagenet.py；用显式类保留相同计算过程。
加载顺序参照 TensorShield 的 get_imagenet_victimnet：先完整加载，再替换最后一层。
VGG 沿用 TSDP 的 7×7 自适应池化，补足 pretrainedmodels 0.7.4 包装中省略的调用。
严格确定性的 CUDA 路径使用同样的区域均值，避开 PyTorch 2.1.1 不支持的池化反向内核。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models as tv_models

from config import Config, load_config
from randomness import preserve_rng_state
from .weights import WEIGHTS, WEIGHTS_ROOT, verify_weight

NUM_CLASSES = {"c10": 10, "c100": 100, "s10": 10, "t200": 200}


class DeterministicAvgPool2d(nn.AdaptiveAvgPool2d):
    """在严格确定性的 CUDA 路径中，直接对各个自适应池化区域取平均。"""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_cuda or not torch.are_deterministic_algorithms_enabled():
            return super().forward(inputs)
        height, width = inputs.shape[-2:]
        out_height, out_width = self.output_size
        if (height, width) == (out_height, out_width):
            return inputs
        if (height, width) == (1, 1):
            return inputs.expand(*inputs.shape[:-2], out_height, out_width)
        rows = []
        for row in range(out_height):
            top = row * height // out_height
            bottom = ((row + 1) * height + out_height - 1) // out_height
            columns = []
            for column in range(out_width):
                left = column * width // out_width
                right = ((column + 1) * width + out_width - 1) // out_width
                columns.append(inputs[..., top:bottom, left:right].mean(dim=(-2, -1)))
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)


class ResNet(nn.Module):
    """保留 ImageNet 的卷积入口，并统一最后分类层的名称。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.last_linear = model.fc

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_linear(torch.flatten(self.avgpool(features), 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits(self.features(inputs))


class VGG(nn.Module):
    """保留两个 4096 维全连接层及不含参数的 7×7 自适应池化。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._features = model.features
        self.avgpool = DeterministicAvgPool2d(model.avgpool.output_size)
        self.linear0 = model.classifier[0]
        self.relu0 = model.classifier[1]
        self.dropout0 = model.classifier[2]
        self.linear1 = model.classifier[3]
        self.relu1 = model.classifier[4]
        self.dropout1 = model.classifier[5]
        self.last_linear = model.classifier[6]

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._features(inputs)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout0(self.relu0(self.linear0(x)))
        return self.linear1(x)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_linear(self.dropout1(self.relu1(features)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits(self.features(inputs))


class MobileNetV2(nn.Module):
    """保留卷积、全局平均池化和分类前的随机失活。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._features = model.features
        self.dropout0 = model.classifier[0]
        self.last_linear = model.classifier[1]

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._features(inputs)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return self.dropout0(torch.flatten(x, 1))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_linear(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits(self.features(inputs))


MODELS = {
    "resnet18": (tv_models.resnet18, ResNet),
    "resnet50": (tv_models.resnet50, ResNet),
    "vgg16_bn": (tv_models.vgg16_bn, VGG),
    "mobilenetv2": (tv_models.mobilenet_v2, MobileNetV2),
}


def build_model(
    name: str,
    dataset: str,
    config: Config | None = None,
    *,
    weights_root: Path = WEIGHTS_ROOT,
) -> nn.Module:
    """返回已加载预训练主体、替换分类头且全部参数可训练的 CPU 模型。

    不传 config 时读取根目录配置。初始化使用该配置的种子，并恢复调用者的
    随机状态，保证同一模型、类别数和种子不受构造顺序影响。
    实验入口仍需调用 seed_everything(config)，统一后续数据加载与训练的随机性。
    """
    if name not in MODELS:
        raise ValueError(f"未知模型：{name}；可选：{', '.join(MODELS)}")
    if dataset not in NUM_CLASSES:
        raise ValueError(f"未知数据集：{dataset}；可选：{', '.join(NUM_CLASSES)}")
    config = load_config() if config is None else config
    spec = WEIGHTS[name]
    path = Path(weights_root) / spec.filename
    verify_weight(path, spec)
    factory, wrapper = MODELS[name]
    with preserve_rng_state(), torch.device("cpu"):
        # 只设置构造模型用到的 CPU 生成器，不触碰尚未初始化的 CUDA 随机状态。
        torch.random.default_generator.manual_seed(config.seed)
        original = factory(weights=None, num_classes=1000)
        state = torch.load(path, map_location="cpu", weights_only=True)
        original.load_state_dict(state, strict=True)
        model = wrapper(original)
        model.last_linear = nn.Linear(model.last_linear.in_features, NUM_CLASSES[dataset])
    return model
