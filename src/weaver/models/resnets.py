import torch.nn as nn
from typing import Type


__all__ = ['PreactResNet', 'WideResNet']


class Basic(nn.Module):
    def __init__(self, cᵢ: int, cₒ: int, s: int):
        super().__init__()

        # residual path
        self.conv1 = nn.Conv2d(cᵢ, cₒ, 3, s, 1, bias=False)
        self.conv2 = nn.Conv2d(cₒ, cₒ, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cᵢ)
        self.bn2 = nn.BatchNorm2d(cₒ)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # shortcut path
        if (cᵢ != cₒ) or (s != 1):
            self.shortcut = nn.Conv2d(cᵢ, cₒ, 1, s, 0, bias=False)

    def forward(self, x):
        z = self.relu1(self.bn1(x))
        if hasattr(self, 'shortcut'):
            x = self.shortcut(z)
        z = self.conv1(z)
        z = self.relu2(self.bn2(z))
        z = self.conv2(z)
        return x + z


class Bottleneck(nn.Module):
    def __init__(self, cᵢ: int, cₒ: int, s: int):
        super().__init__()

        # residual path
        cⱼ = cₒ // 4
        self.conv1 = nn.Conv2d(cᵢ, cⱼ, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(cⱼ, cⱼ, 3, s, 1, bias=False)
        self.conv3 = nn.Conv2d(cⱼ, cₒ, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(cᵢ)
        self.bn2 = nn.BatchNorm2d(cⱼ)
        self.bn3 = nn.BatchNorm2d(cⱼ)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        # shortcut path
        if (cᵢ != cₒ) or (s != 1):
            self.shortcut = nn.Conv2d(cᵢ, cₒ, 1, s, 0, bias=False)

    def forward(self, x):
        z = self.relu1(self.bn1(x))
        if hasattr(self, 'shortcut'):
            x = self.shortcut(z)
        z = self.conv1(z)
        z = self.relu2(self.bn2(z))
        z = self.conv2(z)
        z = self.relu3(self.bn3(z))
        z = self.conv3(z)
        return x + z


def make_layer(B: Type[nn.Module], n: int, cᵢ: int, cₒ: int, s: int):
    layers = [B(cᵢ, cₒ, s)] + [B(cₒ, cₒ, 1) for _ in range(n - 1)]
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, B: Type[nn.Module], n: list[int], c: list[int], num_classes: int):
        super().__init__()

        self.conv = nn.Conv2d(3, c[0], 3, 1, 1, bias=False)
        self.blocks = nn.ModuleList([
            make_layer(B, n[i], c[i], c[i + 1], (1 if i == 0 else 2))
            for i in range(len(n))
        ])
        self.bn = nn.BatchNorm2d(c[-1])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c[-1], num_classes)

        self.initialize_parameters()

    def initialize_parameters(self, a: float = 0, mode: str = 'fan_out'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=a, mode=mode)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.fc(out)


def PreactResNet(depth: int, num_classes: int = 10):
    if depth == 18:
        n = [2, 2, 2, 2]
        c = [64, 64, 128, 256, 512]
        return ResNet(Basic, n, c, num_classes)
    elif depth == 34:
        n = [3, 4, 6, 3]
        c = [64, 64, 128, 256, 512]
        return ResNet(Basic, n, c, num_classes)
    elif depth == 50:
        n = [3, 4, 6, 3]
        c = [64, 256, 512, 1024, 2048]
        return ResNet(Bottleneck, n, c, num_classes)
    elif depth == 101:
        n = [3, 4, 23, 3]
        c = [64, 256, 512, 1024, 2048]
        return ResNet(Bottleneck, n, c, num_classes)
    elif depth == 152:
        n = [3, 8, 36, 3]
        c = [64, 256, 512, 1024, 2048]
        return ResNet(Bottleneck, n, c, num_classes)
    else:
        n = [(depth - 2) // 9] * 3
        c = [16, 64, 128, 256]
        return ResNet(Bottleneck, n, c, num_classes)


def WideResNet(depth: int, width: int, num_classes: int = 10):
    n = [(depth - 4) // 6] * 3
    c = [16, 16 * width, 32 * width, 64 * width]
    return ResNet(Basic, n, c, num_classes)
