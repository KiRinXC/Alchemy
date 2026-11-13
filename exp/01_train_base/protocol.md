# 玩家 A 基础微调协议

实验编号：`01_train_base`。首轮配置：ResNet18 + CIFAR-100 + seed 42。

## 实验目标

使用全部官方有标签训练集，全量微调玩家 A。得到的最佳模型用于后续 Rank 计算、查询伪标签和提供 B 已知位置的参数。

## 模型与数据

| 模型 | 定义 |
|---|---|
| M_0 | `build_model` 构造的训练起点：加载固定版本的 ImageNet 主体，按任务类别数替换分类头，并使用 A 的种子初始化 |
| M_A | 从 M_0 全量微调后，按测试准确率选出的最佳模型 |

首轮 M_0 为 ImageNet 预训练 ResNet18 主体和 seed 42 初始化的 100 类 `last_linear` 分类头。

数据读取自 `dataset/raw/<数据集>/`，训练和评估均使用真实类别标签。

| 数据集 | 类别数 | 训练图片 | 选模图片 | 训练输入 | 评估输入 |
|---|---:|---|---|---|---|
| c10 | 10 | 官方 train 50,000 | 官方 test 10,000 | 3×32×32 | 3×32×32 |
| c100 | 100 | 官方 train 50,000 | 官方 test 10,000 | 3×32×32 | 3×32×32 |
| s10 | 10 | 官方 train 5,000 | 官方 test 8,000 | 3×96×96 | 3×96×96 |
| t200 | 200 | 官方 train 100,000 | 官方 val 10,000 | 3×32×32 | ResNet/MobileNetV2：3×64×64；VGG：3×32×32 |

完整训练集参与梯度更新，不额外划分内部验证集。STL-10 的无标签数据和 Tiny-ImageNet 的无标签 test 不参与本实验。

## 数据增强与预处理

按作者公开 MS 代码执行数据集映射与变换。下表变换后统一执行 `ToTensor → Normalize`，随机水平翻转概率为 0.5。

| 数据集 | 训练变换 | ResNet/MobileNetV2 评估变换 | 归一化 |
|---|---|---|---|
| c10、c100 | RandomCrop(32, padding=4) → RandomHorizontalFlip | 无裁剪或缩放 | CIFAR |
| s10 | RandomResizedCrop(96) → RandomHorizontalFlip | Resize(128) → CenterCrop(96) | ImageNet |
| t200 | RandomCrop(32, padding=4) → RandomHorizontalFlip | 无裁剪或缩放 | CIFAR |

VGG 在所有数据集上均使用训练变换进行测试选模，包括随机裁剪和水平翻转，与作者训练入口一致。

| 归一化 | 均值 | 标准差 |
|---|---|---|
| CIFAR | (0.4914, 0.4822, 0.4465) | (0.2023, 0.1994, 0.2010) |
| ImageNet | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |

## 训练配置

超参数读取 `configs/finetune.toml`；模型、数据集和运行设置读取本实验的 `config.toml`。

| 参数 | 设置 |
|---|---|
| 初始模型 | M_0 |
| 训练范围 | 全部参数，包括主体和分类头 |
| Epochs | 100 |
| Batch size | 64，训练和评估均保留末尾不足一个批次的样本 |
| 优化器 | SGD |
| 初始学习率 | 0.1 |
| Momentum | 0.5 |
| Weight decay | 5e-4 |
| 学习率调度 | StepLR，step_size=60，gamma=0.1 |
| 损失 | Cross Entropy，真实类别标签，无标签平滑 |
| 训练模式 | `train()`，BN statistics 正常更新 |
| 评估模式 | `eval()`，不计算梯度、不更新参数和 BN statistics |
| 数值精度 | FP32，不使用 AMP |

每轮先训练、再评估，最后调用 `scheduler.step()`。第 1–60 轮学习率为 0.1，第 61–100 轮为 0.01，完成全部 100 轮。

## 选模规则

每轮在完整官方测试集上计算 Top-1 准确率与平均交叉熵；t200 使用官方 val。

- 满足 `test_accuracy >= best_accuracy` 时保存 `best.pth`，作为 M_A。
- 准确率相同时保留较晚轮次，不按 loss 打破平局。
- 每轮保存 `last.pth`，用于从最后完成的轮次恢复训练。

结果报告最佳轮和最后一轮的指标。测试集参与选模，报告的测试准确率属于选模成绩，不作为独立留出测试结果。

## 随机性与结果

随机性统一由根目录 `config.toml`、`config.py` 和 `randomness.py` 管理。默认 seed 42，批量种子为 `[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`，开启确定性计算。这里的 seed 属于 A，后续 B 的种子独立设置。

模型初始化、数据增强和训练打乱使用同一运行种子，每轮不重置。训练集按官方样本顺序加载，再由 DataLoader 每轮打乱；评估集不打乱。默认 4 个数据加载进程、2 个 CPU 线程，两个 DataLoader 使用独立生成器，不启用持久 worker。评估结束后恢复训练进程的全局随机状态。

产物按 `<模型>/<数据集>/seed_<种子>/` 分目录保存：

| 目录 | 文件 | 内容 |
|---|---|---|
| `weights/01_train_base/` | `best.pth` | 最佳模型、选模指标与实际运行配置 |
| `weights/01_train_base/` | `last.pth` | 最近完整轮次的模型、优化器、调度器、随机状态及实际运行配置 |
| `results/01_train_base/` | `summary.json` | 最佳轮和最后一轮的训练/测试准确率与 loss、累计耗时（分钟） |
| `results/01_train_base/` | `history.csv` | 每轮的学习率、训练/测试准确率与 loss、耗时（秒） |

准确率以百分数记录，loss 按样本数求平均，累计耗时由逐轮秒数相加得到。恢复训练时核对实际配置，并恢复 Python、NumPy、Torch、CUDA 和两个 DataLoader 的随机状态。

## TensorShield 对照

训练超参数采用论文 §8.1 Configuration（PDF 第 10 页）的设置；数据用途与每轮测试选模规则参照公开 MS 代码。

数据增强采用作者的[数据集映射与变换](https://github.com/suntong30/TensorShield/blob/main/model-stealing/knockoff/datasets/__init__.py)和[微调入口](https://github.com/suntong30/TensorShield/blob/main/model-stealing/knockoff/victim/train.py)。STL-10 的 96×96、Tiny-ImageNet 的训练 32×32 与评估尺寸按公开代码执行，区别于论文列出的 128×128、224×224。

论文与公开代码的 `weight_decay` 均为 5e-4。学习率衰减间隔采用论文的 60 轮；公开代码默认 `StepLR.step_size=30`，可由命令行覆盖，不能据此认定论文实际使用了 30 轮。
