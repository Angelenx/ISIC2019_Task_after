"""
EfficientNet-B3 模型实现，用于 ISIC 2019 皮肤病变多分类（8类）
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
        dropout: 分类头 dropout 概率

    Returns:
        model: nn.Module
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights, dropout=dropout)

    # 替换分类头：forward 里已先做 avgpool + flatten，classifier 只接收 (B, C) 的 2D 张量
    # 因此 classifier 只能是 Dropout + Linear，不能包含 Pool/Flatten
    in_features = model.classifier[1].in_features  # classifier = [Dropout, Linear]
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


if __name__ == '__main__':
    model = build_efficientnet_b3(num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 300, 300)  # EfficientNet-B3 最小输入约 300
    out = model(x)
    print('Output shape:', out.shape)  # (2, 8)
