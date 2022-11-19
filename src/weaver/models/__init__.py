from torchvision.models import get_model
from .resnets import PreactResNet, WideResNet

__all__ = ['get_classifier']


def get_classifier(src: str, name: str, **kwargs):
    src = src.lower()
    if src == 'weaver':
        if name == 'PreactResNet':
            return PreactResNet(**kwargs)
        if name == 'WideResNet':
            return WideResNet(**kwargs)
        raise ValueError(f"Unknown model name: {name}")
    if src == 'torchvision':
        return get_model(name, **kwargs)
    raise ValueError(f"Unknown source: {src}")
