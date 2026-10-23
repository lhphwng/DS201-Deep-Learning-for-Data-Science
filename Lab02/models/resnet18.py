import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = (nn.Conv2d(in_c, out_c, 1, bias=False)
                         if in_c != out_c else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        stage = lambda i, o: nn.Sequential(ResidualBlock(i, o), ResidualBlock(o, o))
        mp = lambda: nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            mp(),
        )
        self.layers = nn.Sequential(
            stage(64, 64), mp(),        # conv2_x
            stage(64, 128), mp(),       # conv3_x
            stage(128, 256), mp(),      # conv4_x
            stage(256, 512),            # conv5_x
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(self.avgpool(x).flatten(1))
