# 03_topk

按 TensorShield 作者排序选择 Top-k，并固定保护分类头 bias。使用 B 独立分类头和 C100 Demo 的 MS 配置，比较 Joint（主结果）、Freeze 与 All-Shield 的 Best Test Accuracy，同时记录 Epoch 100 Accuracy。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | A / 查询 / B 种子、运行阶段的关键 k |
| [b_training.toml](b_training.toml) | B 的独立训练超参数 |
| [protocol.md](protocol.md) | 当前初始化、保护、训练、选模及记录规则 |
| [train.py](train.py) | 读取查询缓存，分阶段训练、选模并汇总 |
| [plot.py](plot.py) | Best 与 Epoch 100 的 Joint / Freeze 双图 |

首轮：ResNet18 + c100，A / B / 查询 seed 42。固定 500 张缓存图片和完整软标签全部训练，使用 `ToTensor → Normalize`。

只读检查：

```bash
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --dry-run
```

首阶段训练 All-Shield 和 Top-10、11、12、17 的 Freeze / Joint，共 9 组：

```bash
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --stage probe
```

检查首阶段结果后，续跑其余 Top-k：

```bash
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --stage full --resume
```

`--all-seeds` 使用本实验声明的 10 个 B 种子。`--resume` 跳过已完成组，中断组从同一初值重训。

| 目录 | 内容 |
|---|---|
| `dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/` | 已有 500 张输入与完整概率，只读复用 |
| `weights/03_topk/author_ms_fixed_bias/resnet18/c100/a_seed_42/b_seed_42/` | B 公共初始化、组级进度及 Best / Last 权重 |
| `results/03_topk/author_ms_fixed_bias/resnet18/c100/a_seed_42/b_seed_42/` | 作者排序、固定 bias 的指标与曲线 |
| `results/03_topk/resnet18/c100/a_seed_42/b_seed_42/` | 已有 35 组历史结果 |

| 结果文件 | 内容 |
|---|---|
| `summary.json` | Best 黑盒基线、扫描范围与临界 k |
| `history.csv` | 每轮软标签 loss、测试准确率、学习率及耗时 |
| `accuracy.tsv` | 每组 Initial / Best / Epoch 100 准确率与保护成本 |
| `protection.tsv` | 各 Top-k 的保护张量、参数量、比例及分类头模式 |
| `topk_accuracy.png` | Joint 主结果与 Freeze 辅助结果的准确率图 |

测试集参与 Best checkpoint 和 Top-k 选择。首阶段的 `observed_min_k` 只对应已测点；完整扫描后 `critical_k` 才表示全局最小 k。
