# 01_train_base

使用全部官方训练数据和真实标签微调玩家 A，每轮用官方测试集选模。最佳模型用于查询软标签，并提供玩家 B 已知位置的参数。

| 文件 | 用途 |
|---|---|
| [train.py](train.py) | 全量训练、测试选模、模型保存和断点恢复 |
| [config.toml](config.toml) | 默认模型、数据集和运行设置 |
| [protocol.md](protocol.md) | 数据用途、训练与选模规则 |
| [../../configs/finetune.toml](../../configs/finetune.toml) | 共用微调超参数 |

默认组合为 `resnet18 + c100`，种子读取根目录配置。CIFAR-100 使用 50,000 张训练图片和 10,000 张测试图片。

```bash
.venv/bin/python -B run/01_train_base.py
.venv/bin/python -B run/01_train_base.py --model resnet18 --dataset c100 --seed 3
.venv/bin/python -B run/01_train_base.py --resume
```

`--all-seeds` 运行全局配置中的全部种子；`--dry-run` 只核对输入。

权重：`weights/exp/01_train_base/<模型>/<数据集>/seed_<种子>/`。

- `best.pth`：测试准确率最高的模型，同分时保存较晚轮次。
- `last.pth`：最近完整轮次的模型、优化器、调度器和随机状态。

结果：`results/exp/01_train_base/<模型>/<数据集>/seed_<种子>/`。

- `summary.json`：最佳轮和最后一轮的训练/测试准确率与 loss，以及训练评估累计分钟数。
- `history.csv`：每轮的学习率、训练/测试 loss、准确率和耗时（秒）。

准确率以百分数记录。配置见本目录和项目根目录，恢复信息随权重保存。
