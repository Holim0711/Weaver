from .wrn28 import WideResNet
from .prn18 import PreActResNet18

__all__ = [
    'get_custom_classifier',
    'WideResNet',
    'PreActResNet18',
]


def get_custom_classifier(name, **kwargs):
    if name.startswith('wide_resnet28_'):
        depth, width = map(int, name[11:].split('_'))
        return WideResNet(depth, width, **kwargs)
    if name == 'preact_resnet18':
        from .prn18 import PreActResNet18
        return PreActResNet18(**kwargs)
    raise ValueError(f"Unsupported model: {name}")
