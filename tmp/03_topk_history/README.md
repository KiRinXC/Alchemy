# Top-k 历史实验

两套已完成的 Top-k 实验，各 35 组。每套的 `code/` 为原代码和协议快照，`weights/` 为保留权重及训练进度，`results/` 为指标和曲线。

| 目录 | 初始化与排序 | 优化器 | 指标 |
|---|---|---|---|
| `a_init_final_accuracy/` | A 的初始分类头、Alchemy Rank | lr=0.1，wd=5e-4 | Epoch 100 Accuracy |
| `author_ms_demo/` | B 独立分类头、作者 Rank、固定 bias | lr=0.01，wd=0 | Best Test Accuracy |
| `shared/exp/01_train_base/train.py` | 两套实验引用的共享训练工具快照 | — | — |

快照保留运行时的配置、路径与校验记录。当前共享输入的位置为：

| 输入 | 当前路径 |
|---|---|
| Victim | `weights/exp/01_train_base/resnet18/c100/seed_42/best.pth` |
| Rank | `results/exp/02_rank/resnet18/c100/seed_42/rank.tsv` |
| 查询缓存 | `dataset/queries/resnet18/c100/a_seed_42/query_seed_42/ratio_0.01/probabilities.pth` |
| A 公共初始化 | `weights/public_init/resnet18/c100/seed_42/init.pth` |

`a_init_final_accuracy/code/` 对应 Git 提交 `c73d3c3`；`author_ms_demo/code/` 保存本次替换前的工作区代码。历史摘要中的相对路径为运行时原路径；保留权重位于各自 `weights/` 下的同名模型、数据集和种子目录。
