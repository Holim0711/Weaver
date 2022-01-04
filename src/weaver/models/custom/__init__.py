import re


def get_custom_model(name, **kwargs):
    if name.startswith('wide_resnet28_'):
        from .wrn28 import WideResNet28
        return WideResNet28(int(name[14:]), **kwargs)
    if name == 'preact_resnet18':
        from .prn18 import PreActResNet18
        return PreActResNet18()
    raise ValueError(f"Unsupported model: {name}")
