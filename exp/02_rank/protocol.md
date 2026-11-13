# TensorShield Rank 复现协议

实验编号：`02_rank`。首轮配置：ResNet18 + CIFAR-100 + seed 42。

计算内部重要性 $G_k$、注意力迁移 $A_k$ 和综合评分 $I_k = G_k A_k$，输出全部候选的排序，并附上 TensorShield 作者提供的排名。

本实验按论文意图修正内部重要性计算，并补全未公开的注意力迁移细节，作为 Alchemy 后续实验的排序依据。具体计算约定如下。

## 模型与数据

| 模型 | 定义 |
|---|---|
| $M_0$ | `build_model` 构造的 A 训练起点：固定 ImageNet 主体和 seed 42 初始化的 100 类分类头 |
| $M_A$ | `weights/01_train_base/resnet18/c100/seed_42/best.pth`，epoch 69，测试选模准确率 61.72% |
| $M_G$ | $M_A$ 的临时副本，仅用于内部重要性计算 |

直接读取 `dataset/splits/c100/seed_42/order.txt`，索引对应 CIFAR-100 官方训练文件顺序。

| 用途 | 图片数量 | 索引范围 |
|---|---:|---|
| 内部重要性 | 45,000 | 固定排列前 90% |
| 注意力迁移 | 5,000 | 固定排列后 10% |

两部分互不重叠，输入均为 `3×32×32`。两阶段复用 `01_train_base` 的变换函数。

CIFAR Normalize 的均值为 `(0.4914, 0.4822, 0.4465)`，标准差为 `(0.2023, 0.1994, 0.2010)`。

## 内部重要性

| 参数 | 设置 |
|---|---|
| 初始模型 | $M_G = \operatorname{copy}(M_A)$ |
| Epochs | 10 |
| Batch size | 128，保留末尾不足一个批次的样本 |
| 优化器 | SGD，lr=0.01，momentum=0.9，weight_decay=0 |
| 损失 | Cross Entropy，真实类别标签 |
| 模型模式 | `eval()`，BN statistics 不更新 |
| 数据变换 | RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → CIFAR Normalize |

全部参数正常参与前向、反向和 SGD 更新；只统计 17 个候选 weight 的分数。

$$
G_k=\frac{1}{T|W_k|}\sum_{t=1}^{T}\sum_{i\in W_k}
\left|g_{k,i}^{(t)}\Delta w_{k,i}^{(t)}\right|.
$$

$g$ 为当前批次平均交叉熵对更新前参数的梯度；$\Delta w$ 为执行 `optimizer.step()` 前后参数的实际差值，包含 momentum 的作用。$M_G$ 连续更新 10 轮，轮次之间不重新加载 $M_A$。

先逐元素计算 $|g\Delta w|$，再对张量内全部参数求和。累计量使用 FP64 求和，全部批次完成后统一除以 $T$ 和参数数量。模型、梯度和更新使用 FP32，不使用 AMP。

$T$ 表示实际优化器更新次数。本配置每轮 352 个批次，10 轮共 3,520 次更新。只保存 $G_k$，不保存 $M_G$ 权重。

## 注意力迁移

固定 $M_0$、$M_A$，保持 `eval()`，不执行优化器更新。对后 5,000 张图片使用 `ToTensor → CIFAR Normalize`，两个模型接收相同输入、相同真实类别和对应模块位置。运行批次大小为 128。

卷积张量采用 `pytorch-grad-cam==1.5.5` 的 GradCAM，通道权重为目标类别 logit 对该通道特征梯度的空间平均：

$$
\alpha_{k,c}=\frac{1}{H_kW_k}\sum_{h,w}\frac{\partial z_y}{\partial F_{k,c}(h,w)}.
$$

| 候选类型 | 用于余弦计算的响应 |
|---|---|
| 空间尺寸至少 2×2 的 Conv | $\operatorname{ReLU}(\sum_c \alpha_c F_c)$，保留该层原始空间分辨率并展平 |
| 空间尺寸为 1×1 的 Conv | 按通道组成向量 $[\operatorname{ReLU}(\alpha_c F_c(1,1))]_c$ |
| `last_linear.weight` | 分类头输入的 512 维正向通道贡献向量 $\mathbf v$ |

分类头输入为池化、展平后的特征 $\mathbf h$，真实类别 $y$ 的 logit 为 $z_y=\sum_i W_{y,i}h_i+b_y$。按通道计算：

$$
v_i(M,x,y)=\operatorname{ReLU}\left(h_i(x)\frac{\partial z_y}{\partial h_i(x)}\right)
=\operatorname{ReLU}\left(h_i(x)W_{y,i}\right),\qquad i=1,\ldots,512.
$$

两个模型使用各自的输入特征和真实类别对应的分类头权重。直接读取线性层权重计算上述等价式；bias 不进入该向量。该定义是 Alchemy 对全连接层的扩展，分数同时反映特征和分类头权重的变化。

以下用 $R_k$ 统称表中的响应。取 ReLU 后、min-max 归一化前的响应；不插值、不对多层响应求平均。通道向量使用两模型对应的通道顺序。

$$
a_k(x,y)=1-\cos\bigl(R_k(M_0,x,y),R_k(M_A,x,y)\bigr).
$$

两份响应均非零时正常计算；均为零时 $a_k=0$；仅一份为零时 $a_k=1$。零响应也计入所属类别的样本数量。

全部 17 个候选均使用后 5,000 张图片，先在每个类别内求平均，再对 100 类等权平均：

$$
A_k=\frac{1}{100}\sum_{c=1}^{100}\frac{1}{N_c}\sum_{x\in D_c}a_k(x,c).
$$

固定子集必须覆盖全部类别；$N_c$ 由现有排列确定，不重新做分层抽样。

## 综合排序

候选集合排除 BatchNorm、Downsample 和 Stem `conv1.weight`，保留 16 个主路径卷积 weight 和 `last_linear.weight`。

$$
I_k=G_kA_k.
$$

按 $I_k$ 降序排列，完全同分时按张量名称排序以保持确定性，保存全部 17 个候选的排名。

作者候选 Rank 取自 TSDP 保存的作者确认结果，已将 17 项名称固化在 `rank.py`，运行时不依赖外部项目。`author_rank` 记录每个候选在作者排序中的位置。

## 随机性、检查与结果

随机性由根目录 `config.py`、`randomness.py` 管理。首轮只运行 seed 42；4 个数据加载进程、2 个 CPU 线程，DataLoader 使用独立生成器，不启用持久 worker。

两项评分前后检查 $M_0$、$M_A$ 的所有参数和缓冲区完全不变。只读取原始数据、已有排列和已完成的 `01_train_base` 产物。

`results/02_rank/resnet18/c100/seed_42/` 仅保存 `intrinsic.tsv`、`attention.tsv` 和 `rank.tsv`，均记录全部 17 个候选。保护集合与向 B 提供哪些参数由后续实验决定。

内部重要性设置参考 [TensorShield 公开实现](https://github.com/suntong30/TensorShield/blob/main/model-stealing/knockoff/adversary/get_importance.py)。实际更新量、统一归一化、注意力数据划分、1×1 通道向量、分类头贡献向量、类别等权与零响应处理按本协议执行。
