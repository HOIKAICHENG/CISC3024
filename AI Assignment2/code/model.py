import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, blocks=(3, 3, 3), num_classes: int = 10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._layer(16, blocks[0], stride=1)
        self.layer2 = self._layer(32, blocks[1], stride=2)
        self.layer3 = self._layer(64, blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.apply(self._init)

    def _layer(self, channels: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, current_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet((3, 3, 3), num_classes)
