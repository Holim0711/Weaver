import torch.nn as nn
import torchvision
from .custom import get_custom_classifier

__all__ = ['get_classifier', 'get_vectorizer']


def get_classifier(src: str, name: str, **kwargs):
    src = src.lower()
    if src == 'weaver':
        return get_custom_classifier(name, **kwargs)
    if src == 'torchvision':
        return torchvision.models.__dict__[name](**kwargs)
    if src == 'lukemelas':
        from efficientnet_pytorch import EfficientNet
        if kwargs.pop('pretrained', False):
            return EfficientNet.from_pretrained(name, **kwargs)
        else:
            return EfficientNet.from_name(name, **kwargs)

    raise ValueError(f"Unknown source: {src}")


def get_vectorizer(src: str, name: str, **kwargs):
    model = get_classifier(src, name, **kwargs)

    candidates = ['fc', '_fc', 'classifier']

    for c in candidates:
        if hasattr(model, c):
            fc = getattr(model, c)
            if isinstance(fc, nn.Linear):
                setattr(model, c, nn.Identity())
                return model
            raise NotImplementedError(type(fc))

    raise ValueError(f"Cannot find {candidates}")
