# Alchemy

Alchemy 研究在公共预训练模型和部分已知权重可用时，如何恢复目标模型的任务能力。
玩家 A 使用 TensorShield，玩家 B 使用相同模型结构，通过查询 A 得到的完整软标签训练替代模型。

当前已完成运行环境、四个原始数据集、数据划分、预训练模型准备，以及 ResNet18 + c100 的首轮玩家 A 微调、张量排序和 Top-k 保护扫描。后续实验采用 `exp/02_rank` 的 Rank。

## 目录

| 路径 | 内容 |
|---|---|
| [config.toml](config.toml) | 全局随机种子和 B 的数据比例 |
| [configs/finetune.toml](configs/finetune.toml) | 共用的微调超参数 |
| [run/](run/) | 实验流程与准备脚本，用于快速复现 |
| [models/](models/README.md) | 共享的模型结构与加载代码 |
| [dataset/raw/](dataset/raw/README.md) | c10、c100、s10、t200 的原始数据、构成和尺寸说明 |
| [dataset/splits/](dataset/splits/README.md) | A/B 数据划分及样本索引 |
| [dataset/queries/](dataset/queries/) | 查询 A 得到的概率数据 |
| [weights/](weights/) | 公共预训练权重和实验权重 |
| [exp/](exp/) | 正式实验代码、配置和协议 |
| [results/](results/) | 实验指标摘要和逐轮训练数据 |
| [tests/](tests/README.md) | 验证结束后清空的临时工作区 |
| [tmp/01_rawrank/](tmp/01_rawrank/README.md) | 作者原始内部重要性算法的临时对照实验与结果 |
| [docs/](docs/) | 论文和参考资料 |
| [AGENTS.md](AGENTS.md) | 项目协作规则 |

## 代码入口

| 文件 | 用途 |
|---|---|
| [run/env/setup.py](run/env/setup.py) | 建立项目运行环境 |
| [run/env/check.py](run/env/check.py) | 检查依赖与 GPU 运算 |
| [run/env/clean.py](run/env/clean.py) | 清理临时验证文件和缓存 |
| [run/dataset/download.py](run/dataset/download.py) | 获取并校验原始数据集 |
| [run/dataset/split.py](run/dataset/split.py) | 保存官方训练池编号及其随机抽样索引 |
| [run/01_train_base.py](run/01_train_base.py) | 复现玩家 A 的基础微调流程 |
| [exp/01_train_base/](exp/01_train_base/README.md) | 全部训练数据微调、官方测试集选模 |
| [exp/02_rank/](exp/02_rank/README.md) | TensorShield 内部重要性、注意力迁移和完整排序 |
| [exp/03_topk/](exp/03_topk/README.md) | Top-k 保护扫描、Freeze / Joint 训练及黑盒阈值对照 |
| [run/models/download.py](run/models/download.py) | 下载并校验 ImageNet 预训练权重 |
| [models/build.py](models/build.py) | 构造模型、加载预训练主体、按全局种子初始化分类头 |
| [models/weights.py](models/weights.py) | 预训练权重的固定版本、存放位置与完整性校验 |
| [config.py](config.py) | 读取全局配置与命令行覆盖值 |
| [randomness.py](randomness.py) | 管理初始化、抽样和数据加载的随机性 |

## 当前配置

环境：WSL2 / Linux x86_64、Python 3.10、PyTorch `2.1.1+cu121`、TorchVision `0.16.1+cu121`、NumPy `1.24.1`、pretrainedmodels `0.7.4`。
依赖声明见 [pyproject.toml](pyproject.toml)，完整版本见 [uv.lock](uv.lock)。

[config.toml](config.toml) 中默认 `seed = 42`，重复实验种子为 `[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]`，B 的数据比例为 `0.01`。

## 资料

- [TensorShield 官方仓库](https://github.com/suntong30/TensorShield)
- [TensorShield 论文](docs/Sun%20等%20-%202025%20-%20TensorShield%20Safeguarding%20On-Device%20Inference%20by%20Shielding%20Critical%20DNN%20Tensors%20with%20TEE.pdf)
- [Notion：Alchemy 科研项目](https://www.notion.so/40e8a5f3d37081c192f2e09b56bb89f8)
- [Notion：TensorShield 阅读年鉴](https://www.notion.so/34c8a5f3d37080dda51cf0950ae69484)
