# Alchemy 实验协议：Top-k 保护集合选择

**实验编号：** `03_topk`

**首轮配置：** ResNet18 + CIFAR-100，A seed 42，B seed 42

## 1. 实验目标

基于 `02_rank` 的重要性排序，依次扩大 Top-k 保护集合，比较两种模型窃取策略：

- **Freeze：** 冻结已暴露的 Victim 可训练参数，仅训练受保护位置。
- **Joint：** 已暴露参数和受保护参数共同训练。

观察不同 Top-k 下玩家 B 的模型恢复准确率，并分别确定两种策略达到 All-Shield 黑盒水平所需的最小保护数量。

## 2. 模型与数据

| 对象 | 来源 |
|---|---|
| Victim $M_A$ | `01_train_base` 的 `best.pth`，Epoch 69，61.72% |
| 公共初始化 $M_0$ | `models/build.py` 构建的 A 原始训练起点 |
| Rank | `02_rank` 的正式候选排序 |
| B 查询数据 | CIFAR-100 官方训练集的固定 1% 子集 |
| 最终评测 | CIFAR-100 官方测试集，10,000 张 |

Rank 路径：

`results/02_rank/<model>/<dataset>/seed_<a_seed>/rank.tsv`

查询索引：

`dataset/splits/<dataset>/seed_<query_seed>/ratio_0.01/b_indices.txt`

首轮使用 500 张查询图片，查询固定 Victim 一次，缓存完整的 100 类概率分布。

所有 Top-k、Freeze、Joint 和 All-Shield 实验共享相同查询数据及软标签，不使用原始真实标签训练 B。

查询、B 训练与评测统一采用 CIFAR-100 测试 Transform：

`ToTensor → Normalize`

B 直接使用缓存的 500 张归一化查询图片及对应完整软标签，不使用 RandomCrop 或 RandomHorizontalFlip。缓存中的图片张量与查询输入完全一致，Freeze、Joint 和 All-Shield 共用同一份缓存。

## 3. Top-k 保护集合

直接读取 `02_rank` 正式候选文件，按照 `rank` 字段升序排列。

设排序包含 $N$ 个候选：

$$
R=(r_1,r_2,\ldots,r_N)
$$

对于 $k=1,\ldots,N$，构造：

$$
H_k=\{r_1,r_2,\ldots,r_k\}
$$

当前 ResNet18+CIFAR-100 的候选数量为 17，因此首轮运行 Top-1～Top-17。

**保护规则：**

1. 每增加一个 k，新增一个排序单位。
2. 当 `last_linear.weight` 进入保护集合时，同时保护 `last_linear.bias`，二者视为一个排序单位。
3. BN、Downsample、Stem 等被 `02_rank` 排除的张量不参与 Top-k 排序。
4. Freeze 和 Joint 使用完全相同的保护集合。
5. 不重新计算 Rank，不在代码中维护张量名称列表。

运行前核对排名连续、名称无重复、候选参数存在且形状匹配。记录所使用的 Rank 文件及 SHA-256。

## 4. 模型初始化及参数训练范围

设 $\theta_0$ 为公共初始化，$\theta_A$ 为 Victim 参数。

对于 Top-k：

$$
\theta_B^{(0)}[H_k]=\theta_0[H_k]
$$

$$
\theta_B^{(0)}[K_k]=\theta_A[K_k]
$$

其中 $K_k$ 为未受保护的可见参数。

| 项目 | Freeze | Joint |
|---|---|---|
| 受保护参数 $H_k$ | 公共初始化，参与训练 | 公共初始化，参与训练 |
| 可见参数 $K_k$ | A 参数，冻结 | A 参数，参与训练 |
| BN affine 参数 | A 参数，冻结 | A 参数，参与训练 |
| BN running buffers | 复制 A，正常更新 | 复制 A，正常更新 |

所有未保护的 BN、Downsample 等状态均按照可见状态处理。

冻结使用 `requires_grad=False`，优化器仅接收可训练参数。模型统一使用 `train()`，不通过模块 `eval()` 实现冻结。

两种策略的区别只在于可见可训练参数是否参与优化。

## 5. 参考基线

### All-Shield：全保护黑盒

- B 从完整公共 $M_0$ 初始化。
- 不复制任何 Victim 参数或 BN buffers。
- 使用相同软标签查询数据训练全部参数。
- 训练配置与 Top-k 完全一致。

All-Shield 只训练一次，作为 Freeze、Joint 共同的黑盒参考。

### No-Shield：无保护参考

直接使用完整 Victim $M_A$，记录准确率 61.72%，不额外训练。

Top-N 仍有被排除的 BN、Downsample 等参数暴露，因此不将 Top-N 等同于 All-Shield。

## 6. 训练配置

**统一读取 `configs/finetune.toml`，直接采用 `01_train_base` 的训练超参数。**

| 参数 | 设置 |
|---|---|
| Epochs | 100 |
| Batch Size | 64 |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Momentum | 0.5 |
| Weight Decay | 5e-4 |
| Scheduler | StepLR |
| Step Size | 60 |
| Gamma | 0.1 |
| Loss | Soft Cross Entropy |
| Softmax Temperature | 1 |
| 数值精度 | FP32 |
| 模型模式 | `train()` |

唯一的监督目标区别是：A 使用真实标签交叉熵，B 使用 Victim 软标签交叉熵。

$$
L_B=-\frac1N\sum_i\sum_c
p_A(c\mid x_i)\log p_B(c\mid x_i)
$$

不针对受保护参数和可见参数设置不同学习率，不额外调整优化器超参数。

所有组均训练完整 100 epochs。训练与评测均使用 batch size 64，保留不足一个批次的末尾样本。训练 DataLoader 使用 0 个 worker、独立的 B seed 生成器；CPU 线程数为 2。固定测试变换只执行一次，图片张量在内存中复用。每轮训练、评测后调用 scheduler.step()，第 1–60 轮学习率为 0.1，第 61–100 轮为 0.01。

## 7. 随机种子与实验执行

首轮固定：

| 随机性来源 | Seed |
|---|---|
| A 训练 | 42 |
| 公共模型初始化 | A 对应的 42 |
| B 查询子集 | 42 |
| B 训练 | 42 |

公共初始化始终对应 A 的实际训练起点。后续改变 `b_seed` 仅改变 B 的训练随机性与数据加载顺序，不重新初始化公共分类头或重新查询 Victim。

正式重复实验使用 B seeds：

`[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

同一 B seed 下各组使用相同公共初始化来源、查询缓存和训练随机设置；同一 Top-k 的 Freeze 与 Joint 初始完整状态相同。每组开始时重新设置 B seed，评测不消耗训练随机状态。不同 Top-k 的混合初始化由各自保护集合决定。

**执行顺序：**

1. 读取并核对 `02_rank`。
2. 固定公共初始化和 Victim。
3. 生成或加载 500 次软标签查询缓存。
4. 训练 All-Shield 黑盒基线。
5. 从 Top-1 到 Top-N，依次执行 Freeze、Joint。
6. 汇总结果并生成准确率曲线。

首轮当前规模：17 个 Top-k × 2 种策略 + 1 个 All-Shield，共 35 次 B 训练。

## 8. 训练记录与评测

每组记录：

| 指标 | 说明 |
|---|---|
| Initial Accuracy | 训练前官方测试集准确率 |
| Train Loss | 每轮软标签交叉熵 |
| Test Accuracy | 每轮官方测试集 Top-1 准确率 |
| Final Accuracy | Epoch 100 的准确率 |
| Protected Parameters | 实际保护参数数量 |
| Protected Ratio | 保护参数量占模型全部可训练参数量的比例 |

主指标固定为 **Epoch 100 的 Accuracy**，不通过测试集选择最佳 Epoch。

All-Shield、Freeze、Joint 采用相同评测方式。

## 9. 黑盒水平判定

设 All-Shield 的准确率为 $\operatorname{Acc}_{BB}$。

首轮将黑盒近似容差设为 1 个百分点：

$$
\epsilon=1\text{ pp}
$$

对于策略 $s$，定义：

$$
k_s^*=\min\{k:
\operatorname{Acc}_s(k)\leq
\operatorname{Acc}_{BB}+\epsilon\}
$$

分别计算：

- $k_{\mathrm{freeze}}^*$
- $k_{\mathrm{joint}}^*$
- $k_{\mathrm{both}}^*$：同一个 k 下两种策略均满足阈值的最小值。

完整扫描所有候选，不假设准确率随 k 增大而严格单调下降。

若 Top-N 仍无法达到黑盒水平，直接记录未达到。

这些结果用于本阶段保护集合选择，官方测试集参与选择，不将其作为独立留出评测。

## 10. 结果与可视化

代码目录：

```text
exp/03_topk/
├── README.md
├── config.toml
├── protocol.md
├── train.py
└── plot.py
```

结果目录：

```text
results/03_topk/
└── resnet18/c100/a_seed_42/
    └── b_seed_42/
        ├── summary.json
        ├── history.csv
        ├── accuracy.tsv
        └── topk_accuracy.png
```

### Top-k 准确率图

`topk_accuracy.png` 为一张上下排列的双子图。

**上图：Freeze**

- 横轴：Top-k，范围 1～N。
- 纵轴：最终 Accuracy (%)。
- 绘制 Freeze 的准确率折线与数据点。
- 标记最小黑盒保护位置。

**下图：Joint**

- 横轴、纵轴与上图一致。
- 绘制 Joint 的准确率折线与数据点。
- 标记最小黑盒保护位置。

**共同要求：**

- 两图使用相同纵轴刻度。
- 绘制 All-Shield 水平参考线。
- 绘制 Victim Accuracy 参考线。
- 记录每个 k 对应的实际保护参数比例。
- 不使用平滑或拟合曲线替代原始结果。

### 结果文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 黑盒基线、临界 k、保护参数比例和输入标识 |
| `history.csv` | 全部训练组的逐轮 Loss 与 Accuracy |
| `accuracy.tsv` | 每组初始/最终准确率、保护比例、与黑盒差值 |
| `topk_accuracy.png` | Freeze / Joint 上下双图 |

为后续 `04` 复用，保留 All-Shield、Freeze 的最小临界 k、Joint 的最小临界 k，以及共同临界 k 下两种策略的 Epoch 100 checkpoint；重复位置只保存一次。未达到阈值时不保存对应临界模型，其余组训练结束后只保留指标与历史。权重位于 `weights/03_topk/<model>/<dataset>/a_seed_<a_seed>/b_seed_<b_seed>/`。

`progress.pth` 保存运行配置与已完成组的记录。按完整训练组恢复：跳过已完成组，中断组从同一初始化重训。恢复时核对 Rank、Victim、公共初始化、查询缓存、超参数、环境和训练代码标识。结果目录保持上述四个文件。

## 11. 验收标准

1. 从 `02_rank` 动态读取候选列表，不硬编码张量名称与数量。
2. 分类头 weight 和 bias 按一个排序单位保护。
3. Freeze 和 Joint 使用相同模型初始化、数据及超参数。
4. 完成 Top-1～Top-N 的两种训练与 All-Shield 基线。
5. 所有组使用 500 次完整软标签查询，不使用 B 的真实标签。
6. 保存逐轮结果、保护参数比例和上下双图。
7. 得到两种策略及共同满足黑盒阈值的最小 k。
8. 不修改已有 `01_train_base`、`02_rank` 和其他仓库。

**实验边界：** `03_topk` 只研究保护数量与 Freeze/Joint 两种训练策略下的模型窃取效果。后续 `04` 固定保护集合，进一步研究参数替换、平均及模型能力恢复方法，符合相同实验条件的已有结果直接复用。
