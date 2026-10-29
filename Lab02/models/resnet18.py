import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 conv để khớp kích thước khi đổi số kênh / stride
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.relu(self.bn_1(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))

        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 21, pool_between_stages: bool = True):
        super().__init__()

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (bs, 64, 112, 112)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        channels = [64, 128, 256, 512]
        layers = []
        in_channels = 64
        for i, out_channels in enumerate(channels):
            if i > 0 and pool_between_stages:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                stride = 1
            else:
                stride = 1 if i == 0 else 2

            layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            in_channels = out_channels

        self.stages = nn.Sequential(*layers)

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, images: torch.Tensor):

        features = self.relu(self.bn_1(self.conv_1(images)))
        features = self.pooling_1(features)

        features = self.stages(features)

        features = self.avg_pooling(features).flatten(1)
        logits = self.output(features)

        return logits
