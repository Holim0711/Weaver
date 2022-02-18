import re


def get_torchssl_model(name, **kwargs):
    if name.startswith('wide_resnet28_'):
        from .wrn28 import WideResNet28
        return WideResNet28(int(name[14:]), **kwargs)
    raise ValueError(f"Unsupported model: {name}")
