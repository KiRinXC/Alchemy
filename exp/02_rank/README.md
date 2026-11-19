# 02_rank

计算 ResNet18 全部 41 个 weight 的内部重要性、注意力迁移和综合排序，并输出 TensorShield 的 17 项候选排序，作为 Alchemy 后续实验的排序依据。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | 模型、打分超参数和运行配置 |
| [protocol.md](protocol.md) | 模型、数据用途、评分公式和排序规则 |
| [intrinsic.py](intrinsic.py) | 在 Victim 副本上计算内部重要性 |
| [attention.py](attention.py) | 比较卷积、BN 注意力与分类头通道贡献，按类别等权平均 |
| [rank.py](rank.py) | 实验入口、完整排名、候选分类和结果保存 |

首轮为 `resnet18 + c100 + seed 42`，使用 epoch 69 的 A 权重。现有 `order.txt` 的前 45,000 项用于内部重要性，后 5,000 项用于注意力迁移。

```bash
.venv/bin/python -B exp/02_rank/rank.py --seed 42 --dry-run
.venv/bin/python -B exp/02_rank/rank.py --seed 42
```

结果位于 `results/exp/02_rank/resnet18/c100/seed_42/`。

| 文件 | 内容 |
|---|---|
| `intrinsic.tsv` | 全部 41 个 weight 的参数数量和 G_k |
| `attention.tsv` | 全部 41 项的 A_k、响应类型（spatial/channel/linear）及双零、单零样本数 |
| `rank_all.tsv` | 全部 41 项的综合排序，`eligible` 标明是否属于保护候选 |
| `rank.tsv` | TensorShield 规则内的 17 项候选排序，供 `03_topk` 使用；`author_rank` 沿用作者的候选内排名 |
| `rank_excluded.tsv` | 其余 24 项的综合排序，`exclusion_reason` 标明 BN、Downsample、Stem；重叠原因用分号连接 |

三张 Rank 表均包含张量名称、参数数量（`parameters`）、G_k、A_k、I_k 和表内排名（`rank`），按 I_k 降序排列，同分按张量名称排序。候选表与排除表互不重叠，合起来恰好覆盖完整表的 41 项。

`--overwrite` 重新计算并替换该次运行的结果。
