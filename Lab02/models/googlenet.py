import torch
import torch.nn as nn


def _cbr(i, o, k, s=1, p=0, use_bn=False):
    layers = [nn.Conv2d(i, o, k, stride=s, padding=p)]
    if use_bn:
        layers.append(nn.BatchNorm2d(o))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4, use_bn=False):
        super().__init__()
        self.b1 = _cbr(in_c, c1, 1, use_bn=use_bn)
        self.b2 = nn.Sequential(_cbr(in_c, c2[0], 1, use_bn=use_bn),
                                _cbr(c2[0], c2[1], 3, p=1, use_bn=use_bn))
        self.b3 = nn.Sequential(_cbr(in_c, c3[0], 1, use_bn=use_bn),
                                _cbr(c3[0], c3[1], 5, p=2, use_bn=use_bn))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                _cbr(in_c, c4, 1, use_bn=use_bn))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, use_bn=False):
        super().__init__()
        I = lambda *a: Inception(*a, use_bn=use_bn)
        pool = lambda: nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.features = nn.Sequential(
            _cbr(3, 64, 7, s=2, p=3, use_bn=use_bn),     # 112x112x64
            pool(),                                       # 56x56x64
            _cbr(64, 64, 1, use_bn=use_bn),
            _cbr(64, 192, 3, p=1, use_bn=use_bn),         # 56x56x192
            pool(),                                       # 28x28x192
            I(192, 64, (96, 128), (16, 32), 32),          # 3a -> 256
            I(256, 128, (128, 192), (32, 96), 64),        # 3b -> 480
            pool(),                                       # 14x14x480
            I(480, 192, (96, 208), (16, 48), 64),         # 4a -> 512
            I(512, 160, (112, 224), (24, 64), 64),        # 4b -> 512
            I(512, 128, (128, 256), (24, 64), 64),        # 4c -> 512
            I(512, 112, (144, 288), (32, 64), 64),        # 4d -> 528
            I(528, 256, (160, 320), (32, 128), 128),      # 4e -> 832
            pool(),                                       # 7x7x832
            I(832, 256, (160, 320), (32, 128), 128),      # 5a -> 832
            I(832, 384, (192, 384), (48, 128), 128),      # 5b -> 1024
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))
