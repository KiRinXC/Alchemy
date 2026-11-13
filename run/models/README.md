# 权重下载

[download.py](download.py) 下载和校验四个 ImageNet 预训练权重，文件保存在 `weights/pre_train/`。

```bash
.venv/bin/python -B run/models/download.py
.venv/bin/python -B run/models/download.py --verify-only
```

可指定模型名称：`resnet18`、`resnet50`、`vgg16_bn`、`mobilenetv2`；默认处理全部四种模型。
