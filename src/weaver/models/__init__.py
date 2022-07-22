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
    raise ValueError(f"Unknown source: {src}")


def get_vectorizer(src: str, name: str, **kwargs):
    model = get_classifier(src, name, **kwargs)

    candidates = ['fc', '_fc', 'classifier']

    for c in candidates:
        if hasattr(model, c):
            fc = getattr(model, c)
            if isinstance(fc, nn.Linear):
                setattr(model, c, nn.Identity())
                return model  # resnet style
            elif isinstance(fc, nn.Sequential):
                if isinstance(fc[-1], nn.Linear):
                    fc[-1] = nn.Identity()
                    return model  # efficientnet style
            raise NotImplementedError(type(fc))

    raise ValueError(f"Cannot find {candidates}")
