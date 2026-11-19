# Lab01 模型窃取基线配置敏感性验证

## 研究问题

本实验以 Alchemy 自己训练的 CIFAR-100 ResNet18 为 Victim，测量分类头初始化、初始学习率（LR）和权重衰减（WD）如何改变 B 的模型窃取（MS）准确率，以及 B 相对全保护黑盒（All-Shield）的优势。

| 问题 | 比较与判读 |
|---|---|
| RQ1 | 在正式 Exp03 的 LR、WD 下，只改分类头对 Top-12 Joint 有多大影响？再看 Top-10～17 的方向是否一致。 |
| RQ2 | 分别改变 LR、WD，绝对准确率和相对本配置黑盒的差距各改变多少？两项同时改变的效果能否由单因素差值相加描述？ |
| RQ3 | 低 LR、零 WD 时，分类头效应与正式 Exp03 优化设置下相差多少？ |

用户提供的早期 Top-12 Joint 为 22.46%，当前正式 Exp03 为 18.80%，相差 3.66 个百分点。早期数字仅作为研究动机；其保护集合和训练条件未经本 Lab 核实。RQ1 以本次受控比较估计分类头效应，并说明能否据此归因历史差异。实验范围限于基线检查，使用现成排名、同一个查询缓存和单一固定种子，不搜索攻击参数或引入其他恢复策略。

## 实验矩阵

A、B、query 的 seed 均为 42。每行包含一组 All-Shield 和八组作者 Top-10～Top-17 Joint。B0 引用当前正式 Exp03 的九组结果，B1～B5 新训练 45 组，每组完整训练 100 轮，共新增 4,500 轮。比较涵盖 54 个配置与场景组合。

| ID | 名称 | 公共初始化中的分类头 | 初始 LR | WD |
|---|---|---|---:|---:|
| B0 | Baseline | B 的独立随机头 | 0.1 | 0.0005 |
| B1 | Head | A 微调前的随机头 | 0.1 | 0.0005 |
| B2 | LR | B 的独立随机头 | 0.01 | 0.0005 |
| B3 | WD | B 的独立随机头 | 0.1 | 0 |
| B4 | LRWD | B 的独立随机头 | 0.01 | 0 |
| B5 | LRWDHead | A 微调前的随机头 | 0.01 | 0 |

B1−B0、B2−B0、B3−B0 分别给出 Head、LR、WD 的单因素差值。B4−B0 给出 LR 和 WD 的组合效应，B5−B4 给出低 LR、零 WD 条件下的 Head 效应。每个差值均在同一保护场景内计算。

## 固定输入与模型

以下路径均相对于 Alchemy 根目录。Exp01、Exp02、Exp03 的代码、配置、权重和结果作为只读输入。实验直接使用本地已有模型和概率缓存，全程不读写 `tmp/` 历史实验产物。

| 输入 | 文件或目录 |
|---|---|
| Victim | `weights/exp/01_train_base/resnet18/c100/seed_42/best.pth` |
| 作者排名 | `results/exp/02_rank/resnet18/c100/seed_42/rank.tsv` |
| 查询索引 | `dataset/splits/c100/seed_42/ratio_0.01/b_indices.txt` |
| 缓存的输入和软标签 | `dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/probabilities.pth` |
| 正式 B 公共初始化 | `weights/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/public_init.pth` |
| B0 的九组原始记录 | `results/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/` |

Victim 为现有 Exp01 的 Best checkpoint，第 69 轮官方测试准确率为 61.72%。所有配置沿用 Exp03 的公开 ImageNet 预训练 ResNet18 主干、100 类任务头及模型结构。A 和张量排名均沿用现有产物。

B0、B2、B3、B4 使用 Exp03 的 B 公共初始化。B 分类头按 Exp03 的 `head_seed` 从模型、数据集、B seed 派生独立随机流，上述四种配置的完整公共初值逐张量相同。

B1、B5 通过 `models/build.py` 及 A seed 42 重建 A 微调开始前的完整训练起点，仅取该起点的 `last_linear.weight` 和 `last_linear.bias`，替换同一 B 公共初始化中的对应张量。其余参数和缓冲区均与 B0 的公共初值相同。公共初始化的构建与训练随机流隔离，每组从同一训练种子和相同 DataLoader 顺序开始。同一类头初始化在所有保护场景中取相同初值，应用各自保护规则后得到该场景的实际训练起点。

## 保护集合和黑盒

从正式排名按 `author_rank` 升序读取 17 个候选 weight，令 $r_i$ 为作者排名第 $i$ 项。八个受测规模组成 $K=\{10,11,12,13,14,15,16,17\}$。Top-$k$ 的保护集合为

$$
H_k=\{r_1,\ldots,r_k\}\cup\{\texttt{last\_linear.bias}\},\quad k\in K.
$$

固定保护的 bias 占 100 个参数，不占作者排名。`last_linear.weight` 在当前作者排序中排第 3，因此八个 Top-k 都保护完整分类头。每个配置在相同 $k$ 时使用相同的 $H_k$。保护比例使用模型全部 11,227,812 个参数为分母，缓冲区不计入分子或分母。

| 场景 | 训练起点与可训练状态 |
|---|---|
| Top-k Joint | 受保护位置保留该配置的公共初值，暴露的参数及 BN running buffers 从固定 Victim checkpoint 复制。所有参数参与联合微调，BN buffers 正常更新。 |
| All-Shield | 整个模型从该配置的公共初始化开始，参数和 BN buffers 均取公共初值。全部参数在相同 500 张软标签数据上使用本配置的 Head、LR 和 WD 训练。 |

每种配置独立确定黑盒。B0 的正式 All-Shield Best 为 17.71%；其余五种配置均各自训练 All-Shield。Top-17 仍有暴露参数，作为单独场景记录。本 Lab 仅运行 Joint 与 All-Shield，不新增 Freeze、多种子、数据增强或 Epoch 消融。

## 训练与选模

数据读取、保护集合生成、模型训练和评估沿用 Exp03 的逻辑。B 的训练集取自 CIFAR-100 官方训练池中已经确定的 500 张图片及对应的 Victim 完整 100 类 Softmax 概率。全部用于软标签训练，不重新抽样或查询，不使用图片的真实标签。每轮使用 CIFAR-100 官方测试集的全部 10,000 张带标签图片评估一次。

| 项目 | 设置 |
|---|---|
| 输入和精度 | `ToTensor` 后按 Exp03 的 CIFAR 均值和标准差归一化，无增强，FP32 |
| 损失 | Soft Cross Entropy，temperature = 1 |
| 优化器 | SGD，momentum = 0.5，初始 LR 和 WD 取矩阵对应值 |
| 调度器 | StepLR，step size = 60，gamma = 0.1 |
| 训练 | 100 轮，batch size = 64，shuffle = True。每轮 8 个 batch，包含末尾 52 张 |
| BN 和随机性 | 训练时 `train()`，正常更新 BN。组间重置 B 训练随机种子，DataLoader 使用独立、同种子的生成器 |
| 每轮顺序 | 训练、完整测试、最后调用 `scheduler.step()`；第 61 轮起 LR 为初始值的十分之一 |

主要指标为第 1～100 轮的最高测试准确率，同分取较晚一轮。训练前的 Initial Accuracy 单独记录，不参与选模。每组保留 Initial、Best、Best Epoch、Epoch 100 Accuracy 及逐轮 Train Loss、Test Accuracy、实际 Learning Rate。官方测试集同时参与 Best Epoch 和 Top-k 的选择，按测试集选模指标报告。

## 指标、因素差值与判定

对配置 $c$、保护规模 $k$，定义绝对准确率、相对本配置黑盒的差距和相对 B0 的差距变化如下。准确率以百分数表示，所有差值均以百分点（pp）表示。

$$
\begin{aligned}
Acc_c(k) &= \operatorname{BestTestAccuracy}_c(k),\\
G_c(k) &= Acc_c(k)-Acc_c(\mathrm{AllShield}),\\
\Delta G_c(k) &= G_c(k)-G_{\mathrm{B0}}(k).
\end{aligned}
$$

$G_c(k)$ 衡量暴露参数及缓冲区相对于全保护带来的准确率差距，允许取负值。对每个 $k$，分别用 $X=Acc$ 和 $X=G$ 计算以下组合差值：

- LR 与 WD 的非加性部分为 $(X_{\mathrm{B4}}-X_{\mathrm{B0}})-(X_{\mathrm{B2}}-X_{\mathrm{B0}})-(X_{\mathrm{B3}}-X_{\mathrm{B0}})$。
- Head 效应随优化配置的变化为 $(X_{\mathrm{B5}}-X_{\mathrm{B4}})-(X_{\mathrm{B1}}-X_{\mathrm{B0}})$。

每种配置分别求扫描范围内首次达到黑盒水平 +1 pp 的 $k$：

$$
k_c^*=\min\{k\in K : Acc_c(k)\leq Acc_c(\mathrm{AllShield})+1\,\mathrm{pp}\}.
$$

如没有受测 $k$ 满足条件，记录“扫描范围内未达到”。依次核对八个点，允许后续点再次超过阈值。结论只适用于 Top-10～17 和本次种子，不能由部分扫描推断全范围最小 $k$ 或最优保护位置。

## 产物与验收

实验实现位于 `lab/01_basecheck/`。结果与权重分别写入 `results/lab/01_basecheck/resnet18/c100/a_seed_42/b_seed_42/` 和 `weights/lab/01_basecheck/resnet18/c100/a_seed_42/b_seed_42/`。每种配置独立存放，B0 的来源指向 Exp03。

| 汇总产物 | 内容 |
|---|---|
| `summary.json` | 六种配置各自的黑盒、范围内阈值结果和因素对照 |
| `accuracy.tsv` | 54 组的 Initial、Best、Best Epoch、Epoch 100 及相对各自黑盒的差距 |
| `history.csv` | 新增 45 组共 4,500 条逐轮记录 |
| `protection.tsv` | 受测保护集合、张量和参数量 |
| `best_accuracy.png` | Top-10～17 上六条 Best 曲线，分别标出六种配置对应的 All-Shield Best |
| `gap_to_all_shield.png` | 六条 $G_c(k)$ 曲线和 +1 pp 阈值线 |
| `delta_gap.png` | 各变更配置相对 B0 的 $\Delta G_c(k)$ |

各图直接使用原始结果，不平滑、不拟合。正式训练前核对六种配置、八个保护集合、公共初始化各张量与分类头来源、B 训练随机状态及各组数据顺序，并确认 B0 对应正式九组记录与同一查询缓存。完成 45 组训练后核对逐轮历史、按同分取晚重算 Best 与 Best Epoch、核对 Epoch 100 和各自黑盒差值。最终根据实测值回答 RQ1～RQ3。
