"""
EfficientNet-B3 模型实现，用于 ISIC 2019 皮肤病变多分类（8 类）。
基于 torchvision 的 efficientnet_b3，仅替换分类头以适配自定义类别数。
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_efficientnet_b3(num_classes=8, pretrained=True, dropout=0.3):
    """
    构建 EfficientNet-B3 分类模型。

    Args:
        num_classes: 分类数量，ISIC2019 为 8 类
        pretrained: 是否使用 ImageNet 预训练权重
        dropout: 分类头中 Dropout 的概率

    Returns:
        model: nn.Module，输入 (B, 3, H, W)，输出 (B, num_classes) logits
    """
    # 可选加载 ImageNet 预训练权重
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights, dropout=dropout)

    # 替换分类头：torchvision 的 forward 里已先做 avgpool + flatten，
    # 因此传入 classifier 的是 2D 张量 (B, C)，classifier 只能是 [Dropout, Linear]
    in_features = model.classifier[1].in_features  # 原最后一层为 Linear(1536, 1000)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


if __name__ == '__main__':
    # 简单前向测试
    model = build_efficientnet_b3(num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 300, 300)  # EfficientNet-B3 最小输入约 300
    out = model(x)
    print('Output shape:', out.shape)  # (2, 8)
