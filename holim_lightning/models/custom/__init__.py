import re


def get_custom_model(name, **kwargs):
    if m1 := re.compile(r'wide_resnet28_(\d+)').match(name):
        from .wrn28 import WideResNet
        return WideResNet(28, int(m1.group(1)), **kwargs)
    raise ValueError(f"Unsupported model: {name}")
