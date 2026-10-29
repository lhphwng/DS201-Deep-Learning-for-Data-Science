import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: tuple, c3: tuple, c4: int):
        super().__init__()

        # nhánh 1: 1x1 conv
        self.branch_1 = ConvBlock(in_channels, c1, kernel_size=1)

        # nhánh 2: 1x1 conv -> 3x3 conv (pad 1)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, c2[0], kernel_size=1),
            ConvBlock(c2[0], c2[1], kernel_size=3, padding=1)
        )

        # nhánh 3: 1x1 conv -> 5x5 conv (pad 2)
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, c3[0], kernel_size=1),
            ConvBlock(c3[0], c3[1], kernel_size=5, padding=2)
        )

        # nhánh 4: 3x3 max pooling (pad 1) -> 1x1 conv
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, c4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
            self.branch_4(x)
        ], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()

        self.conv_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)  # (bs, 64, 112, 112)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # (bs, 64, 56, 56)

        self.conv_2 = nn.Sequential(
            ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 192, kernel_size=3, padding=1)
        )  # (bs, 192, 56, 56)
        self.pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # (bs, 192, 28, 28)

        self.inception_3a = Inception(192, 64, (96, 128), (16, 32), 32)  # 256
        self.inception_3b = Inception(256, 128, (128, 192), (32, 96), 64)  # 480
        self.pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # (bs, 480, 14, 14)

        self.inception_4a = Inception(480, 192, (96, 208), (16, 48), 64)  # 512
        self.inception_4b = Inception(512, 160, (112, 224), (24, 64), 64)  # 512
        self.inception_4c = Inception(512, 128, (128, 256), (24, 64), 64)  # 512
        self.inception_4d = Inception(512, 112, (144, 288), (32, 64), 64)  # 528
        self.inception_4e = Inception(528, 256, (160, 320), (32, 128), 128)  # 832
        self.pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # (bs, 832, 7, 7)

        self.inception_5a = Inception(832, 256, (160, 320), (32, 128), 128)  # 832
        self.inception_5b = Inception(832, 384, (192, 384), (48, 128), 128)  # 1024

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  # (bs, 1024, 1, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, images: torch.Tensor):
        ''''
            images (bs, 3, 224, 224)
        '''

        features = self.pooling_1(self.conv_1(images))  # (bs, 64, 56, 56)
        features = self.pooling_2(self.conv_2(features))  # (bs, 192, 28, 28)

        features = self.inception_3a(features)  # (bs, 256, 28, 28)
        features = self.inception_3b(features)  # (bs, 480, 28, 28)
        features = self.pooling_3(features)  # (bs, 480, 14, 14)

        features = self.inception_4a(features)  # (bs, 512, 14, 14)
        features = self.inception_4b(features)  # (bs, 512, 14, 14)
        features = self.inception_4c(features)  # (bs, 512, 14, 14)
        features = self.inception_4d(features)  # (bs, 528, 14, 14)
        features = self.inception_4e(features)  # (bs, 832, 14, 14)
        features = self.pooling_4(features)  # (bs, 832, 7, 7)

        features = self.inception_5a(features)  # (bs, 832, 7, 7)
        features = self.inception_5b(features)  # (bs, 1024, 7, 7)

        features = self.avg_pooling(features).flatten(1)  # (bs, 1024)
        features = self.dropout(features)
        logits = self.output(features)  # (bs, num_classes)

        return logits
