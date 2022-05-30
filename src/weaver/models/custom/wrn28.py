import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
        self, cᵢ: int, cₒ: int, s: int, leaky_slope: float = 0.,
        activate_before_residual: bool = False
    ):
        super().__init__()

        # Why using `activate_before_residual`
        # → https://github.com/google-research/mixmatch/issues/11
        self.activate_before_residual = activate_before_residual

        # residual path
        self.conv1 = nn.Conv2d(cᵢ, cₒ, 3, s, 1, bias=False)
        self.conv2 = nn.Conv2d(cₒ, cₒ, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cᵢ)
        self.bn2 = nn.BatchNorm2d(cₒ)
        self.relu1 = nn.LeakyReLU(leaky_slope, inplace=True)
        self.relu2 = nn.LeakyReLU(leaky_slope, inplace=True)

        # shortcut path
        if (cᵢ == cₒ) and (s == 1):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(cᵢ, cₒ, 1, s, 0, bias=False)

    def forward(self, x):
        z = self.relu1(self.bn1(x))
        if self.activate_before_residual:
            x = z
        z = self.conv1(z)
        z = self.relu2(self.bn2(z))
        z = self.conv2(z)
        x = self.shortcut(x)
        return x + z


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        width: int = 2,
        num_classes: int = 10,
        leaky_slope: float = 0.,
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        c = [16, 16 * width, 32 * width, 64 * width]

        self.conv0 = nn.Conv2d(3, c[0], 3, 1, 1, bias=False)
        self.block1 = self._make_layer(n, c[0], c[1], 1, leaky_slope,
                                       activate_before_residual=True)
        self.block2 = self._make_layer(n, c[1], c[2], 2, leaky_slope)
        self.block3 = self._make_layer(n, c[2], c[3], 2, leaky_slope)

        self.bn = nn.BatchNorm2d(c[-1])
        self.relu = nn.LeakyReLU(leaky_slope, inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c[-1], num_classes)

        self.init_params(leaky_slope)

    def _make_layer(self, n, cᵢ, cₒ, s, leaky_slope,
                    activate_before_residual=False):
        return nn.Sequential(*[
            BasicBlock(cᵢ, cₒ, s, leaky_slope, activate_before_residual),
            *[BasicBlock(cₒ, cₒ, 1, leaky_slope) for _ in range(1, n)]
        ])

    def init_params(self, leaky_slope):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, leaky_slope, mode='fan_out')
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
