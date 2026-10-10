"""适配 55×55 单通道明场图像的 ResNet-18。"""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, *, initialization: str = "imagenet") -> nn.Module:
    """构建小图分类模型；默认微调 ImageNet 预训练主干。"""
    if num_classes < 2:
        raise ValueError("类别数量至少为 2")
    if initialization not in {"imagenet", "random"}:
        raise ValueError("initialization 必须为 'imagenet' 或 'random'")

    # 先加载预训练主干，再替换尺寸/通道数不兼容的首层与分类头。
    weights = ResNet18_Weights.IMAGENET1K_V1 if initialization == "imagenet" else None
    model = resnet18(weights=weights)

    # 明场图只有一个通道；小图使用 3×3、步长 1 的输入层。
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    # 首层后接 ReLU，使用与 torchvision ResNet 一致的 Kaiming 初始化。
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # 去掉最初的下采样池化，避免 55×55 的细节过早丢失。
    model.maxpool = nn.Identity()

    # 新分类头保留 PyTorch Linear 的默认初始化；预训练主干不重新初始化。
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
