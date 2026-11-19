# TensorShield 关键张量排序复现协议（修订版）

实验编号：`01_rawrank`。首轮配置：ResNet18 + CIFAR-100 + seed 42。

## 实验目标与原则

复现关键张量评分 $I_k=G_kA_k$：$G_k$ 为内部重要性，$A_k$ 为注意力转移。

已有的内部重要性实现保留作者实际数值运算、执行顺序、优化行为和排序逻辑；缺失的注意力转移按论文、作者说明和下述补充约定实现。

全部生成文件位于 `tmp/01_rawrank/`，包括初始化权重、缓存、验证文件和结果。其他目录只读。本实验保留结果，不执行 B0–B4。

## 模型与数据

| 模型 | 定义 |
|---|---|
| $M_0$ | ImageNet 预训练 ResNet18 主体和 seed 42 初始化的 100 类分类头 |
| $M_A$ | `weights/exp/01_train_base/resnet18/c100/seed_42/best.pth`，epoch 69，测试选模准确率 61.72% |
| $M_G$ | $M_A$ 的临时副本，仅用于内部重要性计算 |

公共初始化优先读取项目已有的 `weights/public_init/resnet18/c100/seed_42/init.pth`。该文件不存在时，使用 `models/build.py` 构建，并固定在本实验自己的 `weights/public_init/resnet18/c100/seed_42/init.pth`。加载时核对其与同种子构建结果一致。

复用项目的 `dataset/splits/c100/seed_42/order.txt`：

| 用途 | 数量 | 来源 |
|---|---:|---|
| 内部重要性 | 45,000 | 固定排列前 90% |
| 注意力转移 | 5,000 | 固定排列后 10% |

两个子集互不重叠，均来自官方训练集；A 的原始训练使用完整 50,000 张。首轮 seed 42，全局重复种子为 `[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`。

输入均为 `3×32×32`。CIFAR Normalize 的均值为 `(0.4914, 0.4822, 0.4465)`，标准差为 `(0.2023, 0.1994, 0.2010)`。

## 内部重要性

实现依据：[TensorShield get_importance.py](https://github.com/suntong30/TensorShield/blob/main/model-stealing/knockoff/adversary/get_importance.py)，源码 Git blob 为 `15f764470e304c5bf17385161a7aaaf7d4d10a77`。

| 参数 | 设置 |
|---|---|
| 初始模型 | $M_G=\operatorname{copy}(M_A)$ |
| Epochs | 10 |
| Batch size | 128，保留末尾不足一个批次的样本 |
| 优化器 | SGD，lr=0.01，momentum=0.9，weight_decay=0 |
| 损失 | Cross Entropy，真实标签 |
| 模型模式 | `eval()`，BN running statistics 不更新 |
| 参数更新 | 全部参数正常执行 `optimizer.step()` |
| 数据变换 | RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → CIFAR Normalize |
| 数值精度 | 模型和累计缓冲区均为 FP32，与作者 `zeros_like` 一致 |

每个 batch 顺序固定为前向、反向、SGD 更新、评分累计。评分使用：

$$
\Delta w_{k,i}^{(t)}=-\eta g_{k,i}^{(t)}.
$$

其中 $g$ 为当前批次的梯度，$n_k$ 为张量元素数量。以 $S^{(0)}=0$ 开始：

$$
S_{k,i}^{(t)}=\frac{S_{k,i}^{(t-1)}+|\Delta w_{k,i}^{(t)}|\,|g_{k,i}^{(t)}|}{n_k},
\qquad G_k=\sum_{i=1}^{n_k}S_{k,i}^{(T)}.
$$

保留作者的 `weight_updates.add_`、`weight_importance.addcmul_`、每个 batch 的 `div_(param.numel())`，以及每轮和最终的排序。Momentum 影响优化轨迹，评分仍使用 `-lr × grad`。不改成实际更新量，不在结束时追加统一归一化。

保留所有参数的初始累计槽位，只对名称含 `weight` 的参数累计；最终输出 41 个 weight 的完整 $G_k$ 排序。排序同分时保留作者使用的参数遍历顺序。10 轮共 3,520 次更新，不保存 $M_G$。

固定划分、数据加载器随机生成器、checkpoint 路径与结果存储是 Alchemy 的运行适配；数值计算核心与官方源码逐项对照验证。

## 注意力转移

固定原始 $M_0$、$M_A$，保持 `eval()`，允许计算梯度，不执行优化器更新。使用后 5,000 张图片、真实类别 $y$、batch size 128 和 `ToTensor → CIFAR Normalize`。

两个模型使用相同图片、目标类别和对应模块位置。全部 41 个 weight 均计算注意力：20 个卷积、20 个 BN 和 1 个分类头。

卷积与 BN 采用 `pytorch-grad-cam==1.5.5` 的 GradCAM 通道权重：

$$
\alpha_{k,c}=\frac{1}{H_kW_k}\sum_{h,w}\frac{\partial z_y}{\partial F_{k,c}(h,w)}.
$$

| 张量对应模块 | 余弦计算使用的响应 $R_k$ |
|---|---|
| 空间输出至少 2×2 的卷积或 BN | $\operatorname{ReLU}(\sum_c\alpha_{k,c}F_{k,c})$，保留原始空间尺寸 |
| 空间输出为 1×1 的卷积或 BN | $\operatorname{ReLU}(F_k\odot\partial z_y/\partial F_k)$，保留通道向量 |
| `last_linear.weight` | 分类头的 512 维正向通道贡献向量 |

BN 的扩展由本次协议确认：对 BN 模块自身的原始输出计算响应，在最终候选筛选时排除 BN。原始输出在原地 ReLU 或残差加法之前保存；特征和梯度按模块对应，避免残差分支的回调顺序混淆。

分类头 $z=Wh+b$ 中，$h\in\mathbb R^{512}$，$W\in\mathbb R^{100\times512}$。两个模型使用各自的特征和真实类别权重：

$$
v_{\mathrm{head}}(M,x,y)=\operatorname{ReLU}\bigl(h_M(x)\odot W_{M,y}\bigr).
$$

`last_linear.weight` 使用该向量计算 $A_k$；bias 不独立评分。

每层分别计算 ReLU 后、min-max 归一化前的响应，不缩放、不合并多个层。单样本注意力转移为：

$$
a_k(x,y)=1-\cos\bigl(R_k(M_0,x,y),R_k(M_A,x,y)\bigr).
$$

| 情况 | $a_k$ |
|---|---|
| 两个响应均非零 | 正常计算余弦差异 |
| 两个响应均为零 | 0 |
| 仅一个响应为零 | 1 |

先在每个类别内部平均，再对 100 个类别等权平均：

$$
A_{k,c}=\frac{1}{N_c}\sum_{x\in D_c}a_k(x,c),
\qquad A_k=\frac{1}{100}\sum_{c=1}^{100}A_{k,c}.
$$

固定子集必须覆盖全部类别，不重新抽样。BN、1×1 通道向量、分类头贡献和零向量规则均属于 Alchemy 对未公开部分的补充约定。

## 综合 Rank 与保护集合

$$
I_k=G_kA_k.
$$

按 $I_k$ 降序生成完整 41 项 Rank，完全同分时保留原始 $G_k$ 排名的先后次序。随后排除 BN、Downsample 和 Stem `conv1.weight`，从剩余 17 个候选中选取前 10 个 weight。分类头 weight 入选时，额外纳入 `last_linear.bias`。

作者完整 Rank 取自 TSDP 保存的确认结果，41 个名称固化在本实验代码中。作者 Top-10 取自官方 README，核对其与作者完整 Rank 过滤后的前 10 项一致。记录交集、双方各自包含的张量和完整顺序对照。

## 输出与验收

代码位于 `tmp/01_rawrank/`，首轮结果位于其中的 `results/resnet18/c100/seed_42/`：

| 文件 | 内容 |
|---|---|
| `intrinsic.tsv` | 全部 41 个 weight 的 G_k、参数量和原始排名 |
| `attention.tsv` | 全部 41 项 A_k、响应类型与零向量计数 |
| `rank.tsv` | 完整排名、G_k、A_k、I_k、候选状态和作者排名 |
| `rank_filtered.tsv` | 过滤后的全部 17 个候选及评分，本次排名和作者排名均为候选内的 1–17 |
| `summary.json` | 实际配置、输入标识、Top-10、作者对照和验收记录 |

运行前后检查 $M_0$、$M_A$ 的全部参数与缓冲区；核对输入 checkpoint 和固定划分未被改写。数值核心须与官方源码在相同初值、输入和 SGD 设置下逐项一致。

不保存优化后的临时模型、逐 batch 梯度或 CAM 图片。不修改其他仓库、正式实验和结果目录。缓存与临时验证文件仅在本目录内生成并在验收后清理；实验代码、固定初始化和五份结果保留。
