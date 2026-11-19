# Alchemy 实验协议：Top-k 保护效果

**实验编号：** `03_topk`

**首轮配置：** ResNet18 + CIFAR-100，A / B / 查询 seed 均为 42

## 1. 实验目标

使用 TensorShield 作者 Rank，逐步扩大 Top-k 保护集合，观察 B 的模型窃取准确率是否接近 All-Shield。

B 沿用 A 的优化器与学习率配置，使用自己独立初始化的分类头、500 张无增强查询图片及对应的完整软标签。

- **Joint（主结果）：** 所有可训练参数共同微调。
- **Freeze（辅助对照）：** 冻结暴露的 Victim 可训练参数，只训练受保护位置。

## 2. 模型与数据

| 参数 | 玩家 A | 玩家 B |
|---|---|---|
| 预训练主干 | 公开 ImageNet ResNet18 | 相同公开主干 |
| 100 类分类头 | A 随机初始化 | B 独立随机初始化 |
| 训练数据 | 官方训练集全部 50,000 张 | 已有查询缓存中的 500 张 |
| 训练标签 | 真实标签 | A 输出的完整 100 类 Softmax 概率 |
| 数据增强 | RandomCrop(32, padding=4) + RandomHorizontalFlip | 无 |
| 输入变换 | ToTensor → CIFAR Normalize | 相同归一化 |
| 每轮评测 | 官方测试集 10,000 张 | 相同 |

Normalize：mean = `(0.4914, 0.4822, 0.4465)`，std = `(0.2023, 0.1994, 0.2010)`。输入尺寸为 `3×32×32`。

A 使用已训练的 Victim，Epoch 69，Best Test Accuracy 61.72%。本实验不重新训练 A，不重新计算 Rank。

输入文件：

```text
weights/exp/01_train_base/resnet18/c100/seed_42/best.pth
results/exp/02_rank/resnet18/c100/seed_42/rank.tsv
dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/probabilities.pth
```

B 的 500 张图片全部参与训练，不划分查询验证集，不使用真实标签训练 B。查询、B 训练和测试统一使用 `ToTensor → Normalize`。

B 直接读取缓存的归一化图片与完整概率，保持输入与软标签一一对应；不重新查询，不应用随机裁剪或翻转。

## 3. 初始化与保护集合

### B 的独立分类头

使用 `models/build.py` 加载公开主干，再通过独立随机流重置 B 的分类头 weight 和 bias。

随机流由 `alchemy/03_topk/b_head/<model>/<dataset>/<b_seed>` 的 SHA-256 派生：取前 8 字节的大端整数，保留低 63 位，再设置第 63 位为 1。使用根目录 `preserve_rng_state()` 保存和恢复随机状态；分类头初始化不消耗训练随机流。

A seed 与 B seed 同为 42 时，两者分类头初值也不同。同一 B seed 下，所有 Top-k、Freeze、Joint、All-Shield 共用同一份 B 公共初始化。改变 B seed 不改变 A 或查询数据。

### 作者 Top-k 与固定 bias

从 `rank.tsv` 动态读取 `author_rank` 升序，保留原有 17 项 eligible weight；不使用 Alchemy 的 `rank` 顺序，不硬编码作者张量列表。

令 $r_i$ 为作者第 $i$ 项候选：

$$
H_k=\{r_1,\ldots,r_k\}\cup\{\texttt{last\_linear.bias}\},\qquad k=1,\ldots,17
$$

**所有 Top-k 固定保护分类头 bias，这是与 TSDP `lab/04_tensorshield` 对齐的实验约定。Bias 不占 Top-k 排名，但计入保护参数量。**

| 范围 | 分类头 weight | 分类头 bias | 初始化模式 |
|---|---|---|---|
| Top-1、Top-2 | 从已训练 Victim 复制 | B 自己的随机值 | mixed |
| Top-3～Top-17 | B 自己的随机值 | B 自己的随机值 | replace |

分类头 weight 在当前作者候选排序中为第 3 项。候选排除 BatchNorm、Downsample 和 Stem `conv1.weight`。

| 项目 | Freeze | Joint |
|---|---|---|
| 受保护参数 | B 公共初始化，训练 | B 公共初始化，训练 |
| 未受保护参数 | 从 A 复制，冻结 | 从 A 复制，训练 |
| BN running buffers | 从 A 复制，正常更新 | 从 A 复制，正常更新 |
| 模型模式 | `train()` | `train()` |

Freeze 通过 `requires_grad=False` 冻结暴露参数；两种策略的初始完整状态相同。保护比例的分母为全部 11,227,812 个参数，分子包含固定保护的 100 个 bias 参数，BN buffers 不计入参数量。

## 4. 训练与选模

A 的配置为 `configs/finetune.toml`，B 的配置为 `exp/03_topk/b_training.toml`。程序核对两者除损失函数外的训练超参数一致。

| 参数 | 玩家 A | 玩家 B |
|---|---|---|
| Learning Rate | 0.1 | 0.1 |
| Weight Decay | 5e-4 | 5e-4 |
| Optimizer | SGD | SGD |
| Momentum | 0.5 | 0.5 |
| Epochs | 100 | 100 |
| Batch Size | 64 | 64 |
| StepLR step size / gamma | 60 / 0.1 | 60 / 0.1 |
| Loss | Cross Entropy | Soft Cross Entropy |
| Checkpoint | Best Test Accuracy | Best Test Accuracy |

B 的 Softmax temperature 为 1，使用 FP32、0 个 DataLoader worker、2 个 CPU threads。

$$
L_B=-\frac{1}{N}\sum_i\sum_c p_A(c\mid x_i)\log p_B(c\mid x_i)
$$

每轮打乱全部 500 张图片，保留末尾不足 64 张的 batch，共 8 batches。训练和评测均使用 batch size 64。第 1～60 轮 lr = 0.1，第 61～100 轮 lr = 0.01；每轮结束调用 `scheduler.step()`。

每组开始通过根目录 `seed_everything()` 设置 B seed，DataLoader 使用同种子的独立生成器。评测保存和恢复训练随机状态，不在 batch 或 epoch 内重新设种子。

**主指标为 Epoch 1～100 的 Best Test Accuracy，同分保留较晚 Epoch。** 初始准确率不参与选模；同时保存 Epoch 100 Accuracy 及完整训练历史。

**官方测试集参与 Best checkpoint 和 Top-k 选择，结果不是独立留出测试。**

## 5. 基线、扫描与黑盒判定

| 基线 | 定义 |
|---|---|
| All-Shield | 使用 B 自己的完整公共初始化，不复制任何 A 参数或 buffer，训练全部参数 |
| No-Shield | 直接评测已有 Victim，61.72%，不训练 |

All-Shield 与 Freeze / Joint 使用相同的查询缓存、训练超参数和 B seed。Top-17 仍暴露 BN、Downsample、Stem，因此不等同于 All-Shield。

每个 B seed 运行 **All-Shield + Top-1～Top-17 的 Freeze / Joint，共 35 组，每组 100 轮**。首轮 B seed 42；可选重复种子为 `[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`。

正式黑盒基线取本次重新训练的、同一 B seed 的 **All-Shield Best Accuracy**，容差为 1 个百分点：

$$
k_s^*=\min\{k:\operatorname{BestAcc}_s(k)\leq\operatorname{BestAcc}_{BB}+1\text{ pp}\}
$$

分别统计 Freeze、Joint 和两者同时满足的最小 k。不假设准确率单调，不因结果未达到预期而改变配置；完整扫描后 `critical_k=null` 表示未达到阈值。

## 6. 结果与权重

```text
results/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/
weights/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/
```

同一正式实验编号只保存当前协议对应的数据。

| 结果文件 | 内容 |
|---|---|
| `accuracy.tsv` | Initial / Best / Best Epoch / Epoch 100、保护量、比例及与黑盒的差值 |
| `history.csv` | 每组每轮的 soft CE、测试准确率、学习率、耗时 |
| `protection.tsv` | 各 Top-k 与 All-Shield 的保护张量、数量、比例及分类头模式 |
| `summary.json` | 黑盒基线、临界 k、输入标识与保留权重 |
| `topk_accuracy.png` | Joint / Freeze 的 Best 与 Epoch 100 曲线 |

准确率为百分数，差值为百分点，`protected_ratio` 为 0～1。曲线展示原始数据，不平滑，并注明测试集参与选模。

每组训练保存 `<group>.best.pth` 和 `<group>.last.pth`；完整扫描结束后保留 All-Shield 与最终临界组的 Best / Last，全部 35 组指标与历史完整保留。

B 公共初始化保存为 `public_init.pth`。`progress.pth` 保存实际配置及组级进度；中断组从同一初值重训，已完成组跳过。
