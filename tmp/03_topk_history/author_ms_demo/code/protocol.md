# Alchemy 实验协议：TensorShield 标准 MS Top-k 复现

**实验编号：** `03_topk`

**首轮配置：** ResNet18 + CIFAR-100，A / B / 查询 seed 均为 42

## 1. 实验目标

使用 TensorShield 作者 Rank 和 C100 Demo 的标准 MS 配置，逐步扩大保护集合，观察 B 的准确率是否接近 All-Shield。

- **Joint（主结果）：** 所有可训练参数共同微调。
- **Freeze（辅助对照）：** 冻结暴露的 Victim 可训练参数，只训练受保护位置。

不修改 Victim、Rank，不加入权重平均、插值或额外恢复算法。

## 2. 模型与查询数据

| 对象 | 定义 |
|---|---|
| Victim $M_A$ | `01_train_base` 的 `best.pth`，Epoch 69，61.72% |
| B 公共初始化 $M_B^0$ | 相同的公开 ImageNet ResNet18 主干 + B 独立初始化的 100 类分类头 |
| Rank | `02_rank/rank.tsv` 的 `author_rank` 升序，共 17 项 |
| B 训练集 | 已缓存的 500 张 CIFAR-100 查询图片及其完整 100 类 Softmax 概率 |
| 每轮评测集 | CIFAR-100 官方测试集，10,000 张 |

输入文件：

```text
weights/01_train_base/resnet18/c100/seed_42/best.pth
results/02_rank/resnet18/c100/seed_42/rank.tsv
dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/probabilities.pth
```

500 张全部训练，不划分查询验证集，不用真实标签训练 B。查询、B 训练、测试均使用确定性变换：

`ToTensor → CIFAR Normalize`

Normalize：mean = `(0.4914, 0.4822, 0.4465)`，std = `(0.2023, 0.1994, 0.2010)`。输入尺寸为 `3×32×32`。

B 直接读取缓存中的归一化图片与完整概率，输入和软标签一一对应；不重新查询，也不使用随机裁剪或翻转。

## 3. 初始化与固定 bias 保护

### B 的独立分类头

通过 `models/build.py` 加载公开主干，再用独立随机流重置 B 的分类头 weight 和 bias。B 不读取 A 的初始分类头。

随机流由以下字符串的 SHA-256 派生：

`alchemy/03_topk/b_head/<model>/<dataset>/<b_seed>`

取前 8 字节的大端整数，保留低 63 位，再设置第 63 位为 1。派生种子与项目 32 位训练种子的数值范围分离。使用根目录 `preserve_rng_state()`，只重置 CPU 分类头初始化随机流；训练随机状态不被消耗。派生种子随 B 权重记录。

同一 B seed 下，所有 Top-k、Freeze、Joint、All-Shield 共用同一份 $M_B^0$。更改 B seed 会改变 B 的分类头及训练顺序，不改变 A 或查询数据。A seed 与 B seed 同为 42 时，两者分类头初值也不同。

### Top-k 保护集合

令 $r_i$ 为 `author_rank` 第 $i$ 项：

$$
H_k=\{r_1,\ldots,r_k\}\cup\{\text{last_linear.bias}\},\qquad k=1,\ldots,17
$$

**固定 bias 是与 TSDP `lab/04_tensorshield` 对齐的实验约定。Bias 不占 Top-k 排名，但计入保护参数量。**

| 范围 | 分类头 weight | 分类头 bias | 初始化模式 |
|---|---|---|---|
| Top-1、Top-2 | 从已训练 Victim 复制 | 保留 B 自己的随机值 | mixed |
| Top-3～Top-17 | 保留 B 自己的随机值 | 保留 B 自己的随机值 | replace |

分类头 weight 在当前作者候选排序中为第 3 项，代码动态读取，不硬编码作者名称列表。

对于每个 Top-k：受保护参数来自 $M_B^0$；其余参数及全部 BN buffers 来自 $M_A$。候选仍排除 BatchNorm、Downsample 和 Stem `conv1.weight`。

| 项目 | Freeze | Joint |
|---|---|---|
| 受保护参数 | 训练 | 训练 |
| 暴露的 Victim 参数 | `requires_grad=False` | 训练 |
| BN running buffers | 从 A 复制，训练时正常更新 | 从 A 复制，训练时正常更新 |
| 模型模式 | `train()` | `train()` |

两种策略初始完整状态相同。参数比例的分母为模型全部可训练参数量 11,227,812，不包含 BN buffers；分子包含固定保护的 100 个 bias 参数。

## 4. 训练与选模

B 独立读取 `b_training.toml`，A 的 `configs/finetune.toml` 不变。

| 参数 | 设置 |
|---|---|
| Epochs / Batch Size | 100 / 64 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum / Weight Decay | 0.5 / 0 |
| Scheduler | StepLR，step size = 60，gamma = 0.1 |
| Loss | Soft Cross Entropy |
| Temperature / Precision | 1 / FP32 |
| DataLoader / CPU threads | 0 workers / 2 threads |

$$
L_B=-\frac{1}{N}\sum_i\sum_c p_A(c\mid x_i)\log p_B(c\mid x_i)
$$

每轮打乱全部 500 张图片，保留末尾不足 64 张的 batch，共 8 batches。训练和评测均使用 batch size 64。第 1～60 轮 lr = 0.01，第 61～100 轮 lr = 0.001；每轮结束调用 `scheduler.step()`。

每组开始通过根目录 `seed_everything()` 设置 B seed，DataLoader 使用同种子的独立生成器。评测保存和恢复训练随机状态，不在 batch 或 epoch 内重新设种子。

**主指标为 Epoch 1～100 的 Best Test Accuracy，同分保留较晚 Epoch。** 初始准确率不参与选模，同时记录 Epoch 100 Accuracy 及完整训练历史。

**官方测试集参与 Best checkpoint 和 Top-k 选择，结果不是独立留出测试。**

实现依据：TensorShield `model-stealing-demo/knockoff/adversary/train.py` 的 C100 Demo 默认设置和提前构造的 SGD；其 `weight_decay=0`，不采用 `utils/model.py` 在未传优化器时的 `5e-4` 默认值。本实验固定 lr = 0.01，不混用主扫描脚本的 lr = 0.1。Best 及同分规则对应 Demo `knockoff/utils/model.py`。

## 5. 基线、阶段与黑盒判定

| 基线 | 定义 |
|---|---|
| All-Shield | 完整使用 $M_B^0$，不复制任何 A 的参数或 buffer，训练全部参数 |
| No-Shield | 直接评测已有 Victim，61.72%，不训练 |

同一 B seed 的正式 All-Shield 只训练一次。Top-17 仍暴露 BN、Downsample、Stem，不等同于全保护。

执行分为两个阶段：

1. `probe`：All-Shield，以及作者 Top-10、11、12、17 的 Freeze / Joint，共 9 组。
2. `full`：检查首阶段结果后，沿用其进度，补齐其余 Top-k，共计 35 组。

不会从 `probe` 自动启动 `full`。重复 B seeds 为 `[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`，单次默认读取根目录 seed 42。

正式黑盒基线取同一实验设置、同一 B seed 的 **All-Shield Best Accuracy**，容差为 1 个百分点：

$$
k_s^*=\min\{k:\operatorname{BestAcc}_s(k)\leq\operatorname{BestAcc}_{BB}+1\text{ pp}\}
$$

分别统计 Freeze、Joint 和两者同时满足的最小 k；不假设准确率单调，不因结果未达预期而调整配置。

首阶段仅记录已测 k 中的 `observed_min_k`，`critical_k` 保持 `null`。完整扫描后才报告全局最小 k；此时 `null` 表示全部候选均未达到阈值。

## 6. 结果与权重

结果与权重目录：

```text
results/03_topk/author_ms_fixed_bias/resnet18/c100/a_seed_42/b_seed_42/
weights/03_topk/author_ms_fixed_bias/resnet18/c100/a_seed_42/b_seed_42/
```

| 结果文件 | 内容 |
|---|---|
| `accuracy.tsv` | 初始、Best、Best Epoch、Epoch 100 准确率，保护量、比例及 Best 与黑盒的差值 |
| `history.csv` | 每组每轮的 soft CE、测试准确率、学习率、耗时 |
| `protection.tsv` | 全部 Top-k 与 All-Shield 的保护张量、数量、比例、分类头模式 |
| `summary.json` | Best 黑盒基线、扫描范围、临界 k、输入标识与保留权重 |
| `topk_accuracy.png` | Joint 主结果、Freeze 辅助结果；Best 与 Epoch 100 同时展示 |

准确率单位为百分数，差值为百分点，`protected_ratio` 为 0～1。图中首阶段只画已测点；完整扫描展示原始折线，不平滑。图注明测试集参与选模。

每组保存 `<group>.best.pth` 和 `<group>.last.pth`。首阶段保留全部组的权重；完整扫描结束后保留 All-Shield 和最终临界组的 Best / Last。全部组指标与历史始终保留。

B 公共初始化保存为权重目录下的 `public_init.pth`。`progress.pth` 保存实际配置及组级进度；中断的组从同一初值重训，已完成组跳过。

已有 35 组结果及权重保留原路径，不纳入本次基线；对应配置与代码通过 Git 提交 `c73d3c3` 查阅。
