# 02_rank

计算 TensorShield 候选张量的内部重要性、注意力迁移和完整排序，作为 Alchemy 后续实验的排序依据。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | 模型、打分超参数和运行配置 |
| [protocol.md](protocol.md) | 模型、数据用途、评分公式和排序规则 |
| [intrinsic.py](intrinsic.py) | 在 Victim 副本上计算内部重要性 |
| [attention.py](attention.py) | 比较卷积注意力与分类头通道贡献，按类别等权平均 |
| [rank.py](rank.py) | 实验入口、综合排名和结果保存 |

首轮为 `resnet18 + c100 + seed 42`，使用 epoch 69 的 A 权重。现有 `order.txt` 的前 45,000 项用于内部重要性，后 5,000 项用于注意力迁移。

```bash
.venv/bin/python -B exp/02_rank/rank.py --seed 42 --dry-run
.venv/bin/python -B exp/02_rank/rank.py --seed 42
```

结果位于 `results/02_rank/resnet18/c100/seed_42/`。

| 文件 | 内容 |
|---|---|
| `intrinsic.tsv` | 17 个候选的参数数量和 G_k |
| `attention.tsv` | A_k、响应类型（spatial/channel/linear）及双零、单零样本数 |
| `rank.tsv` | 17 个候选的 G_k、A_k、I_k、排名及作者排名 |

`--overwrite` 重新计算并替换该次运行的结果。
