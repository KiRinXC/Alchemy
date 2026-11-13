# ResNet18 + CIFAR-100

结构与可训练状态取自 Alchemy 的 `build_model("resnet18", "c100")`。输入为 `3×32×32`，分类头为 `last_linear`，输出 100 类 logits；文中尺寸均省略 batch 维。TensorShield 列对照 TSDP 同模型、同数据集的既有保护掩码。

模型共有 **122 个状态单元（unit 0–121）**：62 个可训练参数张量、60 个 BN 缓冲区，可训练参数总量为 **11,227,812**。

TSDP 的 TensorShield 保护 **11 个单元**，编号为 `6,12,18,24,30,36,48,54,60,120,121`；保护参数 **1,009,764 / 11,227,812（8.9934%）**。全部 BN 和 downsample 状态均未保护。

## 逐单元对照

每个 unit 单独一行，编号沿用 TSDP 的 `state_dict` 顺序；所有单元的名称、顺序和张量尺寸均已与 Alchemy 模型核对。

“本单元可训练”表示该行张量属于 `nn.Parameter` 且 `requires_grad=True`，对应模型构造后默认未冻结的状态。BN 的 `weight`、`bias` 可训练；`running_mean`、`running_var`、`num_batches_tracked` 是缓冲区，训练时可以更新，但不通过梯度优化。BN 模块通过其 `weight` 和 `bias` 包含可训练参数。

“候选排名”采用 TSDP 过滤后的 17 个权重排序，`—` 表示未进入这一候选序列。当前保护 Top-1 至 Top-10，并配套保护 `last_linear.bias`。“TensorShield 保护”记录掩码中的权重可见性，“本单元可训练”记录梯度优化属性。

| unit 编号 | 单元名称 | 类型 | 张量尺寸 | 本单元可训练 | 候选排名 | TensorShield 保护 |
|---:|---|---|---|---|---:|---|
| 0 | `conv1.weight` | 参数 | `64×3×7×7` | 是 | — | 否 |
| 1 | `bn1.weight` | 参数 | `64` | 是 | — | 否 |
| 2 | `bn1.bias` | 参数 | `64` | 是 | — | 否 |
| 3 | `bn1.running_mean` | 缓冲区 | `64` | 否 | — | 否 |
| 4 | `bn1.running_var` | 缓冲区 | `64` | 否 | — | 否 |
| 5 | `bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 6 | `layer1.0.conv1.weight` | 参数 | `64×64×3×3` | 是 | Top-4 | **是** |
| 7 | `layer1.0.bn1.weight` | 参数 | `64` | 是 | — | 否 |
| 8 | `layer1.0.bn1.bias` | 参数 | `64` | 是 | — | 否 |
| 9 | `layer1.0.bn1.running_mean` | 缓冲区 | `64` | 否 | — | 否 |
| 10 | `layer1.0.bn1.running_var` | 缓冲区 | `64` | 否 | — | 否 |
| 11 | `layer1.0.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 12 | `layer1.0.conv2.weight` | 参数 | `64×64×3×3` | 是 | Top-8 | **是** |
| 13 | `layer1.0.bn2.weight` | 参数 | `64` | 是 | — | 否 |
| 14 | `layer1.0.bn2.bias` | 参数 | `64` | 是 | — | 否 |
| 15 | `layer1.0.bn2.running_mean` | 缓冲区 | `64` | 否 | — | 否 |
| 16 | `layer1.0.bn2.running_var` | 缓冲区 | `64` | 否 | — | 否 |
| 17 | `layer1.0.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 18 | `layer1.1.conv1.weight` | 参数 | `64×64×3×3` | 是 | Top-1 | **是** |
| 19 | `layer1.1.bn1.weight` | 参数 | `64` | 是 | — | 否 |
| 20 | `layer1.1.bn1.bias` | 参数 | `64` | 是 | — | 否 |
| 21 | `layer1.1.bn1.running_mean` | 缓冲区 | `64` | 否 | — | 否 |
| 22 | `layer1.1.bn1.running_var` | 缓冲区 | `64` | 否 | — | 否 |
| 23 | `layer1.1.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 24 | `layer1.1.conv2.weight` | 参数 | `64×64×3×3` | 是 | Top-5 | **是** |
| 25 | `layer1.1.bn2.weight` | 参数 | `64` | 是 | — | 否 |
| 26 | `layer1.1.bn2.bias` | 参数 | `64` | 是 | — | 否 |
| 27 | `layer1.1.bn2.running_mean` | 缓冲区 | `64` | 否 | — | 否 |
| 28 | `layer1.1.bn2.running_var` | 缓冲区 | `64` | 否 | — | 否 |
| 29 | `layer1.1.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 30 | `layer2.0.conv1.weight` | 参数 | `128×64×3×3` | 是 | Top-2 | **是** |
| 31 | `layer2.0.bn1.weight` | 参数 | `128` | 是 | — | 否 |
| 32 | `layer2.0.bn1.bias` | 参数 | `128` | 是 | — | 否 |
| 33 | `layer2.0.bn1.running_mean` | 缓冲区 | `128` | 否 | — | 否 |
| 34 | `layer2.0.bn1.running_var` | 缓冲区 | `128` | 否 | — | 否 |
| 35 | `layer2.0.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 36 | `layer2.0.conv2.weight` | 参数 | `128×128×3×3` | 是 | Top-6 | **是** |
| 37 | `layer2.0.bn2.weight` | 参数 | `128` | 是 | — | 否 |
| 38 | `layer2.0.bn2.bias` | 参数 | `128` | 是 | — | 否 |
| 39 | `layer2.0.bn2.running_mean` | 缓冲区 | `128` | 否 | — | 否 |
| 40 | `layer2.0.bn2.running_var` | 缓冲区 | `128` | 否 | — | 否 |
| 41 | `layer2.0.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 42 | `layer2.0.downsample.0.weight` | 参数 | `128×64×1×1` | 是 | — | 否 |
| 43 | `layer2.0.downsample.1.weight` | 参数 | `128` | 是 | — | 否 |
| 44 | `layer2.0.downsample.1.bias` | 参数 | `128` | 是 | — | 否 |
| 45 | `layer2.0.downsample.1.running_mean` | 缓冲区 | `128` | 否 | — | 否 |
| 46 | `layer2.0.downsample.1.running_var` | 缓冲区 | `128` | 否 | — | 否 |
| 47 | `layer2.0.downsample.1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 48 | `layer2.1.conv1.weight` | 参数 | `128×128×3×3` | 是 | Top-7 | **是** |
| 49 | `layer2.1.bn1.weight` | 参数 | `128` | 是 | — | 否 |
| 50 | `layer2.1.bn1.bias` | 参数 | `128` | 是 | — | 否 |
| 51 | `layer2.1.bn1.running_mean` | 缓冲区 | `128` | 否 | — | 否 |
| 52 | `layer2.1.bn1.running_var` | 缓冲区 | `128` | 否 | — | 否 |
| 53 | `layer2.1.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 54 | `layer2.1.conv2.weight` | 参数 | `128×128×3×3` | 是 | Top-10 | **是** |
| 55 | `layer2.1.bn2.weight` | 参数 | `128` | 是 | — | 否 |
| 56 | `layer2.1.bn2.bias` | 参数 | `128` | 是 | — | 否 |
| 57 | `layer2.1.bn2.running_mean` | 缓冲区 | `128` | 否 | — | 否 |
| 58 | `layer2.1.bn2.running_var` | 缓冲区 | `128` | 否 | — | 否 |
| 59 | `layer2.1.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 60 | `layer3.0.conv1.weight` | 参数 | `256×128×3×3` | 是 | Top-9 | **是** |
| 61 | `layer3.0.bn1.weight` | 参数 | `256` | 是 | — | 否 |
| 62 | `layer3.0.bn1.bias` | 参数 | `256` | 是 | — | 否 |
| 63 | `layer3.0.bn1.running_mean` | 缓冲区 | `256` | 否 | — | 否 |
| 64 | `layer3.0.bn1.running_var` | 缓冲区 | `256` | 否 | — | 否 |
| 65 | `layer3.0.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 66 | `layer3.0.conv2.weight` | 参数 | `256×256×3×3` | 是 | Top-11 | 否 |
| 67 | `layer3.0.bn2.weight` | 参数 | `256` | 是 | — | 否 |
| 68 | `layer3.0.bn2.bias` | 参数 | `256` | 是 | — | 否 |
| 69 | `layer3.0.bn2.running_mean` | 缓冲区 | `256` | 否 | — | 否 |
| 70 | `layer3.0.bn2.running_var` | 缓冲区 | `256` | 否 | — | 否 |
| 71 | `layer3.0.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 72 | `layer3.0.downsample.0.weight` | 参数 | `256×128×1×1` | 是 | — | 否 |
| 73 | `layer3.0.downsample.1.weight` | 参数 | `256` | 是 | — | 否 |
| 74 | `layer3.0.downsample.1.bias` | 参数 | `256` | 是 | — | 否 |
| 75 | `layer3.0.downsample.1.running_mean` | 缓冲区 | `256` | 否 | — | 否 |
| 76 | `layer3.0.downsample.1.running_var` | 缓冲区 | `256` | 否 | — | 否 |
| 77 | `layer3.0.downsample.1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 78 | `layer3.1.conv1.weight` | 参数 | `256×256×3×3` | 是 | Top-17 | 否 |
| 79 | `layer3.1.bn1.weight` | 参数 | `256` | 是 | — | 否 |
| 80 | `layer3.1.bn1.bias` | 参数 | `256` | 是 | — | 否 |
| 81 | `layer3.1.bn1.running_mean` | 缓冲区 | `256` | 否 | — | 否 |
| 82 | `layer3.1.bn1.running_var` | 缓冲区 | `256` | 否 | — | 否 |
| 83 | `layer3.1.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 84 | `layer3.1.conv2.weight` | 参数 | `256×256×3×3` | 是 | Top-16 | 否 |
| 85 | `layer3.1.bn2.weight` | 参数 | `256` | 是 | — | 否 |
| 86 | `layer3.1.bn2.bias` | 参数 | `256` | 是 | — | 否 |
| 87 | `layer3.1.bn2.running_mean` | 缓冲区 | `256` | 否 | — | 否 |
| 88 | `layer3.1.bn2.running_var` | 缓冲区 | `256` | 否 | — | 否 |
| 89 | `layer3.1.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 90 | `layer4.0.conv1.weight` | 参数 | `512×256×3×3` | 是 | Top-12 | 否 |
| 91 | `layer4.0.bn1.weight` | 参数 | `512` | 是 | — | 否 |
| 92 | `layer4.0.bn1.bias` | 参数 | `512` | 是 | — | 否 |
| 93 | `layer4.0.bn1.running_mean` | 缓冲区 | `512` | 否 | — | 否 |
| 94 | `layer4.0.bn1.running_var` | 缓冲区 | `512` | 否 | — | 否 |
| 95 | `layer4.0.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 96 | `layer4.0.conv2.weight` | 参数 | `512×512×3×3` | 是 | Top-13 | 否 |
| 97 | `layer4.0.bn2.weight` | 参数 | `512` | 是 | — | 否 |
| 98 | `layer4.0.bn2.bias` | 参数 | `512` | 是 | — | 否 |
| 99 | `layer4.0.bn2.running_mean` | 缓冲区 | `512` | 否 | — | 否 |
| 100 | `layer4.0.bn2.running_var` | 缓冲区 | `512` | 否 | — | 否 |
| 101 | `layer4.0.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 102 | `layer4.0.downsample.0.weight` | 参数 | `512×256×1×1` | 是 | — | 否 |
| 103 | `layer4.0.downsample.1.weight` | 参数 | `512` | 是 | — | 否 |
| 104 | `layer4.0.downsample.1.bias` | 参数 | `512` | 是 | — | 否 |
| 105 | `layer4.0.downsample.1.running_mean` | 缓冲区 | `512` | 否 | — | 否 |
| 106 | `layer4.0.downsample.1.running_var` | 缓冲区 | `512` | 否 | — | 否 |
| 107 | `layer4.0.downsample.1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 108 | `layer4.1.conv1.weight` | 参数 | `512×512×3×3` | 是 | Top-14 | 否 |
| 109 | `layer4.1.bn1.weight` | 参数 | `512` | 是 | — | 否 |
| 110 | `layer4.1.bn1.bias` | 参数 | `512` | 是 | — | 否 |
| 111 | `layer4.1.bn1.running_mean` | 缓冲区 | `512` | 否 | — | 否 |
| 112 | `layer4.1.bn1.running_var` | 缓冲区 | `512` | 否 | — | 否 |
| 113 | `layer4.1.bn1.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 114 | `layer4.1.conv2.weight` | 参数 | `512×512×3×3` | 是 | Top-15 | 否 |
| 115 | `layer4.1.bn2.weight` | 参数 | `512` | 是 | — | 否 |
| 116 | `layer4.1.bn2.bias` | 参数 | `512` | 是 | — | 否 |
| 117 | `layer4.1.bn2.running_mean` | 缓冲区 | `512` | 否 | — | 否 |
| 118 | `layer4.1.bn2.running_var` | 缓冲区 | `512` | 否 | — | 否 |
| 119 | `layer4.1.bn2.num_batches_tracked` | 缓冲区 | `标量` | 否 | — | 否 |
| 120 | `last_linear.weight` | 参数 | `100×512` | 是 | Top-3 | **是** |
| 121 | `last_linear.bias` | 参数 | `100` | 是 | — | **是（分类头配套）** |

## 不含可训练参数的计算操作

以下操作按前向执行顺序逐次列出，共 28 次调用，均不含可训练参数，也没有独立的 `state_dict` unit 编号。同一 BasicBlock 的 ReLU 调用两次，分别列出；恒等捷径直接复用块输入，投影捷径使用上表中的 `downsample.0` 卷积和 `downsample.1` BN。

| 计算操作 | 用途 | 输出尺寸 | 包含可训练参数 |
|---|---|---|---|
| `relu` | ReLU 激活 | `64×16×16` | 否 |
| `maxpool` | 3×3 最大池化，stride=2，padding=1 | `64×8×8` | 否 |
| `layer1.0.relu（第1次）` | ReLU 激活 | `64×8×8` | 否 |
| `layer1.0.add` | 主分支与捷径分支逐元素相加 | `64×8×8` | 否 |
| `layer1.0.relu（第2次）` | ReLU 激活 | `64×8×8` | 否 |
| `layer1.1.relu（第1次）` | ReLU 激活 | `64×8×8` | 否 |
| `layer1.1.add` | 主分支与捷径分支逐元素相加 | `64×8×8` | 否 |
| `layer1.1.relu（第2次）` | ReLU 激活 | `64×8×8` | 否 |
| `layer2.0.relu（第1次）` | ReLU 激活 | `128×4×4` | 否 |
| `layer2.0.add` | 主分支与捷径分支逐元素相加 | `128×4×4` | 否 |
| `layer2.0.relu（第2次）` | ReLU 激活 | `128×4×4` | 否 |
| `layer2.1.relu（第1次）` | ReLU 激活 | `128×4×4` | 否 |
| `layer2.1.add` | 主分支与捷径分支逐元素相加 | `128×4×4` | 否 |
| `layer2.1.relu（第2次）` | ReLU 激活 | `128×4×4` | 否 |
| `layer3.0.relu（第1次）` | ReLU 激活 | `256×2×2` | 否 |
| `layer3.0.add` | 主分支与捷径分支逐元素相加 | `256×2×2` | 否 |
| `layer3.0.relu（第2次）` | ReLU 激活 | `256×2×2` | 否 |
| `layer3.1.relu（第1次）` | ReLU 激活 | `256×2×2` | 否 |
| `layer3.1.add` | 主分支与捷径分支逐元素相加 | `256×2×2` | 否 |
| `layer3.1.relu（第2次）` | ReLU 激活 | `256×2×2` | 否 |
| `layer4.0.relu（第1次）` | ReLU 激活 | `512×1×1` | 否 |
| `layer4.0.add` | 主分支与捷径分支逐元素相加 | `512×1×1` | 否 |
| `layer4.0.relu（第2次）` | ReLU 激活 | `512×1×1` | 否 |
| `layer4.1.relu（第1次）` | ReLU 激活 | `512×1×1` | 否 |
| `layer4.1.add` | 主分支与捷径分支逐元素相加 | `512×1×1` | 否 |
| `layer4.1.relu（第2次）` | ReLU 激活 | `512×1×1` | 否 |
| `avgpool` | 自适应平均池化到 1×1 | `512×1×1` | 否 |
| `flatten` | 保留 batch 维，将其余维度展平 | `512` | 否 |

完整前向计算共 69 次操作：20 次卷积、20 次 BN、17 次 ReLU、8 次残差相加、1 次最大池化、1 次自适应平均池化、1 次 Flatten、1 次全连接。状态单元数与计算调用次数分别统计。

## 对照位置

- [Alchemy 模型结构与分类头](../../models/build.py)
- [TSDP 单元编号定义](../../../TSDP/exp/MS/train_surrogate/defense/resnet18.py)
- [TSDP TensorShield 候选排序与保护集合](../../../TSDP/exp/MS/train_surrogate/selector/tensorshield.py)
- [TSDP TensorShield 掩码构造](../../../TSDP/exp/MS/train_surrogate/defense/tensorshield.py)
- [TSDP 实际保护掩码](../../../TSDP/weights/MS/surrogate/resnet18/c100/tensorshield/protection_mask.pt)
- [TSDP 保护数量与参数比例记录](../../../TSDP/results/MS/resnet18/c100/tensorshield/metrics.json)
