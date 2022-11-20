from torchvision.models import get_model
from .resnets import PreactResNet, WideResNet

__all__ = ['get_classifier']


def get_classifier(src: str, name: str, **kwargs):
    src = src.lower()
    if src == 'weaver':
        if name.startswith('preact_resnet'):
            depth = int(name[13:])
            return PreactResNet(depth, **kwargs)
        if name.startswith('wide_resnet'):
            depth, width = map(int, name[11:].split('_'))
            return WideResNet(depth, width, **kwargs)
        raise ValueError(f"Unknown model name: {name}")
    if src == 'torchvision':
        return get_model(name, **kwargs)
    raise ValueError(f"Unknown source: {src}")
