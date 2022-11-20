import torch.nn as nn


class BasicBlock(nn.Module):
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


class WideResNet(nn.Module):
    def __init__(self, depth: int, width: int, num_classes: int = 10):
        super().__init__()

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        c = [16, 16 * width, 32 * width, 64 * width]

        self.conv0 = nn.Conv2d(3, c[0], 3, 1, 1, bias=False)
        self.block1 = self._make_layer(n, c[0], c[1], 1)
        self.block2 = self._make_layer(n, c[1], c[2], 2)
        self.block3 = self._make_layer(n, c[2], c[3], 2)

        self.bn = nn.BatchNorm2d(c[-1])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c[-1], num_classes)

        self.init_params()

    def _make_layer(self, n, cᵢ, cₒ, s):
        return nn.Sequential(*[
            BasicBlock(cᵢ, cₒ, s),
            *[BasicBlock(cₒ, cₒ, 1) for _ in range(1, n)]
        ])

    def init_params(self, a=0.0):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = self.flatten(out)
        return self.fc(out)
