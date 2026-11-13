# 共享模型

| 文件 | 用途 |
|---|---|
| [build.py](build.py) | 模型结构、预训练加载和分类头初始化 |
| [weights.py](weights.py) | 预训练权重的固定版本、存放位置与完整性校验 |

```python
from config import load_config
from randomness import seed_everything
from models.build import build_model

config = load_config()
seed_everything(config)
model = build_model("resnet18", "c100", config)
```

模型名称：`resnet18`、`resnet50`、`vgg16_bn`、`mobilenetv2`。
数据集名称：`c10`、`c100`、`s10`、`t200`，分类数分别为 10、100、10、200。
模型通过 `last_linear` 访问最后分类层，输出分类分数；初始化使用根目录的全局种子。
