import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
        self, cᵢ: int, cₒ: int, s: int,
        dropout: float = 0.,
        negative_slope: float = 0.,
        activate_before_residual: bool = False
    ):
        super().__init__()

        # Why we use activate_before_residual?
        # [1] https://github.com/brain-research/realistic-ssl-evaluation/blob/master/lib/networks.py
        # [2] https://github.com/google-research/mixmatch/issues/11
        self.activate_before_residual = activate_before_residual

        # residual path
        self.conv0 = nn.Conv2d(cᵢ, cₒ, 3, s, 1, bias=False)
        self.conv1 = nn.Conv2d(cₒ, cₒ, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(cᵢ)
        self.bn1 = nn.BatchNorm2d(cₒ)
        self.relu0 = nn.LeakyReLU(negative_slope, inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope, inplace=True)
        self.drop = nn.Dropout(dropout)

        # shortcut path
        if (cᵢ == cₒ) and (s == 1):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(cᵢ, cₒ, 1, s, 0, bias=False)

    def forward(self, x):
        z = self.relu0(self.bn0(x))
        if self.activate_before_residual:
            x = z
        z = self.conv0(z)
        z = self.relu1(self.bn1(z))
        z = self.conv1(self.drop(z))
        x = self.shortcut(x)
        return x + z


def _make_layer(n: int, cᵢ: int, cₒ: int, s: int, **kwargs):
    abr = kwargs.pop('activate_before_residual', False)
    layers = [BasicBlock(cᵢ, cₒ, s, activate_before_residual=abr, **kwargs)]
    layers += [BasicBlock(cₒ, cₒ, 1, **kwargs) for _ in range(n - 1)]
    return nn.Sequential(*layers)


class WideResNet(nn.Module):
    def __init__(self, depth: int, width: int, num_classes: int = 10, **kwargs):
        super().__init__()

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        c = [16, 16 * width, 32 * width, 64 * width]
        negative_slope = kwargs.get('negative_slope', 0.)

        self.conv = nn.Conv2d(3, c[0], 3, 1, 1, bias=False)
        self.block1 = _make_layer(n, c[0], c[1], 1, activate_before_residual=True, **kwargs)
        self.block2 = _make_layer(n, c[1], c[2], 2, **kwargs)
        self.block3 = _make_layer(n, c[2], c[3], 2, **kwargs)
        self.bn = nn.BatchNorm2d(c[3])
        self.relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c[3], num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, negative_slope, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = self.flatten(out)
        return self.fc(out)


def WideResNet28(width, **kwargs):
    return WideResNet(28, width, **kwargs)
