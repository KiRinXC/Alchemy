# 原始数据集

本目录保存四个数据集的原始图片、标签和类别信息。图片尺寸均为高 × 宽。

| 目录 | 数据集 | 类别数 | 数据组成 | 图片总数 | 原始尺寸 |
|---|---|---|---|---:|---|
| [c10](c10/) | CIFAR-10 | 10 | 训练 50,000；测试 10,000 | 60,000 | 32 × 32，RGB |
| [c100](c100/) | CIFAR-100 | 100 个细类、20 个粗类 | 训练 50,000；测试 10,000 | 60,000 | 32 × 32，RGB |
| [s10](s10/) | STL-10 | 有标签部分为 10 类 | 训练 5,000；测试 8,000；无标签 100,000 | 113,000 | 96 × 96，RGB |
| [t200](t200/) | Tiny-ImageNet-200 | 200 | 训练 100,000；验证 10,000；测试 10,000（无标签） | 120,000 | 64 × 64，RGB 或灰度 |

## 文件位置

| 数据目录 | 主要文件及用途 |
|---|---|
| [c10/cifar-10-batches-py/](c10/cifar-10-batches-py/) | `data_batch_1`～`data_batch_5`：训练数据，每份 10,000 张；`test_batch`：测试数据；`batches.meta`：类别名称 |
| [c100/cifar-100-python/](c100/cifar-100-python/) | `train`：训练数据；`test`：测试数据；`meta`：细类和粗类名称 |
| [s10/stl10_binary/](s10/stl10_binary/) | `train_X.bin`、`train_y.bin`：训练图片与标签；`test_X.bin`、`test_y.bin`：测试图片与标签；`unlabeled_X.bin`：无标签图片；`class_names.txt`：类别名称；`fold_indices.txt`：10 个训练子集的索引，每个子集 1,000 张 |
| [t200/tiny-imagenet-200/](t200/tiny-imagenet-200/) | `train/<类别>/images/`：训练图片，每类 500 张；`val/images/`：验证图片，每类 50 张；`val/val_annotations.txt`：验证标签；`test/images/`：测试图片；`wnids.txt`、`words.txt`：类别 ID 与名称 |

各数据集目录中的 `source.json` 是数据来源记录。
