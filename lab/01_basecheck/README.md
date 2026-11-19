# Lab01 模型窃取基线配置敏感性验证

在固定的 ResNet18、CIFAR-100 和 500 张查询图片上，比较分类头初始化、学习率和权重衰减对 Top-10～17 Joint 准确率的影响。每种配置使用自己的 All-Shield 作参照。B0 读取正式 Exp03 的九组记录，B1～B5 共训练 45 组。

| 文件 | 用途 |
|---|---|
| [protocol.md](protocol.md) | 研究问题、控制变量、初始化、训练、选模和指标定义 |
| [config.toml](config.toml) | 固定种子、扫描范围与六组配置 |
| [train.py](train.py) | 运行前检查、训练和指标汇总 |
| [plot.py](plot.py) | 绘制绝对准确率、相对黑盒差距和因素差值 |

在 Alchemy 根目录先检查配置和初始化，再运行完整训练。

```bash
.venv/bin/python -B lab/01_basecheck/train.py --dry-run
.venv/bin/python -B lab/01_basecheck/train.py
```

中断后运行 `.venv/bin/python -B lab/01_basecheck/train.py --resume`，已完成的组直接复用，未完成的组从相同初值重新训练。单独重绘可运行 `.venv/bin/python -B lab/01_basecheck/plot.py`。

[结果目录](../../results/lab/01_basecheck/resnet18/c100/a_seed_42/b_seed_42/)包含六组对照汇总、保护集合、逐轮日志及三张图。Best Accuracy 取官方测试集上 100 轮的最高值。首次达到阈值的 Top-k 也使用该测试集，仅在本次扫描的八个点中比较。

[权重目录](../../weights/lab/01_basecheck/resnet18/c100/a_seed_42/b_seed_42/)按新增配置保存训练进度与 checkpoint。每种配置完成扫描后保留 All-Shield 的 Best / Last，以及范围内首个满足阈值的 Top-k Joint 的 Best / Last；若范围内无命中，只保留 All-Shield 的两份训练权重。B0 权重沿用正式 Exp03。
