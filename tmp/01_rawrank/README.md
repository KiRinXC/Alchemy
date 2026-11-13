# 01_rawrank

临时对照实验：复现 TensorShield 原始内部重要性实现，补全 41 个 weight 的注意力转移、综合排序和 Top-10 对照。

| 文件 | 用途 |
|---|---|
| [config.toml](config.toml) | 模型、打分与运行配置 |
| [protocol.md](protocol.md) | 数据、公式、特殊层规则和验收要求 |
| [intrinsic.py](intrinsic.py) | 作者原始的逐 batch 更新和归一化 |
| [attention.py](attention.py) | 卷积、BN 和分类头的注意力转移 |
| [rank.py](rank.py) | 实验入口、完整排名和作者对照 |
| `weights/public_init/` | 本实验固定的公共初始化 |
| `results/resnet18/c100/seed_42/` | 首轮评分和排序结果 |

在 Alchemy 根目录运行：

```bash
.venv/bin/python -B tmp/01_rawrank/rank.py --seed 42
```

`--dry-run` 核对输入，`--overwrite` 重新计算并替换本实验结果。

| 结果文件 | 内容 |
|---|---|
| `intrinsic.tsv` | 41 个 weight 的原始 G_k 和排名 |
| `attention.tsv` | 41 项 A_k、响应类型、零向量统计 |
| `rank.tsv` | 完整综合 Rank、G_k、A_k、I_k、候选状态和作者排名 |
| `rank_filtered.tsv` | 排除 BN、Downsample 和 Stem 后的 17 项 Rank；本次和作者排名均从 1 重新编号 |
| `summary.json` | 实际配置、Top-10、作者对照及验收记录 |

本实验的初始化、结果、缓存和临时验证文件均放在当前目录内。
