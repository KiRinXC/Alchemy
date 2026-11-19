# 03_topk

从 `02_rank/rank.tsv` 读取候选，扫描全部 Top-k，比较 Freeze、Joint 与 All-Shield 的 Epoch 100 准确率，确定达到黑盒阈值的最小保护数量。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | A、查询种子、B 重复种子及运行设置 |
| [protocol.md](protocol.md) | 初始化、保护、训练、阈值与记录规则 |
| [train.py](train.py) | 固定查询缓存、训练全部组并汇总结果 |
| [plot.py](plot.py) | 绘制 Freeze / Joint 上下双图 |

首轮为 ResNet18 + c100，A seed 42、查询 seed 42、B seed 42。500 张图片统一使用 `ToTensor → Normalize`，缓存图片与完整软标签直接配对训练，不做随机增强。

```bash
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --dry-run
.venv/bin/python -B exp/03_topk/train.py --b-seed 42
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --resume
```

`--all-seeds` 运行本实验配置中的全部 B 种子。恢复以完整训练组为单位：跳过已完成组，中断的组从同一初始化重新训练。

| 目录 | 内容 |
|---|---|
| `dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/` | 500 张实际查询输入、100 类完整概率及索引 |
| `weights/public_init/resnet18/c100/seed_42/` | A 的固定公共初始化 |
| `weights/03_topk/resnet18/c100/a_seed_42/b_seed_42/` | 组级进度、All-Shield 与最终临界 Top-k 的 Epoch 100 权重 |
| `results/03_topk/resnet18/c100/a_seed_42/b_seed_42/` | 首轮指标和曲线 |

| 结果文件 | 内容 |
|---|---|
| `summary.json` | 黑盒基线、三个临界 k、对应保护比例和输入标识 |
| `history.csv` | 全部训练组每轮的软标签 loss、测试准确率、学习率和耗时 |
| `accuracy.tsv` | 每组初始与最终准确率、保护参数量、比例及与黑盒差值 |
| `topk_accuracy.png` | Freeze / Joint 的原始准确率曲线与参考线 |

`protected_ratio` 以 0–1 表示，准确率与 `gap_to_all_shield` 以百分数、百分点表示。临界 k 为 `null` 表示扫描范围内未达到阈值。
