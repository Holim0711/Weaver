# original: https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cᵢ, cₒ, stride=1, dropout=0., norm_layer=None, relu_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if relu_layer is None:
            relu_layer = nn.ReLU

        self.conv0 = nn.Conv2d(cᵢ, cₒ, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = nn.Conv2d(cₒ, cₒ, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = norm_layer(cᵢ)
        self.bn1 = norm_layer(cₒ)

        self.relu0 = relu_layer()
        self.relu1 = relu_layer()
        self.drop = nn.Dropout(dropout)

        self.convdim = None
        if cᵢ != cₒ:
            self.convdim = nn.Conv2d(cᵢ, cₒ, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        a = self.relu0(self.bn0(x))
        z = self.conv0(a)
        z = self.relu1(self.bn1(z))
        z = self.conv1(self.drop(z))
        if self.convdim is not None:
            x = self.convdim(a)
        return z + x


def _make_layer(n, cᵢ, cₒ, stride=1, **kwargs):
    layers = [BasicBlock(cᵢ, cₒ, stride=stride, **kwargs)]
    layers += [BasicBlock(cₒ, cₒ, stride=1, **kwargs) for i in range(1, n)]
    return nn.Sequential(*layers)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth, width,
                 block_dropout=0., dense_dropout=0.,
                 norm_layer=None, relu_layer=None):
        super().__init__()

        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        c = [16, 16 * width, 32 * width, 64 * width]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if relu_layer is None:
            relu_layer = nn.ReLU

        self.conv = nn.Conv2d(3, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = _make_layer(
            n, c[0], c[1], stride=1, dropout=block_dropout, norm_layer=norm_layer, relu_layer=relu_layer)
        self.block2 = _make_layer(
            n, c[1], c[2], stride=2, dropout=block_dropout, norm_layer=norm_layer, relu_layer=relu_layer)
        self.block3 = _make_layer(
            n, c[2], c[3], stride=2, dropout=block_dropout, norm_layer=norm_layer, relu_layer=relu_layer)
        self.bn = norm_layer(c[3])
        self.relu = relu_layer()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dense_dropout)
        self.fc = nn.Linear(c[3], num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(self.drop(out))


def build_wide_resnet28(name, num_classes, **kwargs):
    if name == "wide_resnet28_2":
        depth, width = 28, 2
    elif name == 'wide_resnet28_8':
        depth, width = 28, 8
    else:
        raise ValueError(f"Unsupported model: {name}")
    return WideResNet(num_classes, depth, width, **kwargs)
