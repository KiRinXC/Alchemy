# 03_topk

按 TensorShield 作者 Rank 和固定分类头 bias 规则扫描 Top-1～Top-17。B 使用独立随机分类头、500 张无增强查询图片及完整软标签，优化器超参数与 A 相同。比较 Joint（主结果）、Freeze 和 All-Shield 的 Best Test Accuracy。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | A / 查询 / B 种子及运行设置 |
| [b_training.toml](b_training.toml) | B 的训练超参数与软标签损失 |
| [protocol.md](protocol.md) | 初始化、保护、训练、选模及记录规则 |
| [train.py](train.py) | 训练全部 35 组并汇总 |
| [plot.py](plot.py) | Joint / Freeze 的 Best 与 Epoch 100 双图 |

首轮：ResNet18 + c100，A / B / 查询 seed 42。

```bash
.venv/bin/python -B exp/03_topk/train.py --b-seed 42 --dry-run
.venv/bin/python -B exp/03_topk/train.py --b-seed 42
```

`--resume` 跳过已完成组，中断组从同一初值重训；`--all-seeds` 使用本实验声明的 10 个 B 种子。

| 目录 | 内容 |
|---|---|
| `dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/` | 500 张固定输入与完整软标签 |
| `weights/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/` | B 公共初始化、组级进度及 Best / Last 权重 |
| `results/exp/03_topk/resnet18/c100/a_seed_42/b_seed_42/` | 当前协议的指标与曲线 |

| 结果文件 | 内容 |
|---|---|
| `summary.json` | Best 黑盒基线与临界 k |
| `history.csv` | 每轮软标签 loss、测试准确率、学习率及耗时 |
| `accuracy.tsv` | 每组 Initial / Best / Epoch 100 准确率与保护成本 |
| `protection.tsv` | 保护张量、参数量、比例及分类头模式 |
| `topk_accuracy.png` | 完整 Top-k 准确率曲线 |

测试集参与 Best checkpoint 和 Top-k 选择。
