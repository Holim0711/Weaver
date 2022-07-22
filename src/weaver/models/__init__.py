import torch.nn as nn
import torchvision
from .custom import get_custom_classifier

__all__ = ['get_classifier', 'get_featurizer', 'FEATURE_DIMENSION']


def get_classifier(src: str, name: str, **kwargs):
    src = src.lower()
    if src == 'weaver':
        return get_custom_classifier(name, **kwargs)
    if src == 'torchvision':
        return torchvision.models.__dict__[name](**kwargs)
    raise ValueError(f"Unknown source: {src}")


def get_featurizer(src: str, name: str, **kwargs):
    model = get_classifier(src, name, **kwargs)

    candidates = ['fc', 'classifier']

    for c in candidates:
        if hasattr(model, c):
            fc = getattr(model, c)
            if isinstance(fc, nn.Linear):
                setattr(model, c, nn.Identity())
                return model  # resnet, ...
            elif isinstance(fc, nn.Sequential):
                if isinstance(fc[-1], nn.Linear):
                    fc[-1] = nn.Identity()
                    return model  # vgg, efficientnet, ...
            raise NotImplementedError(type(fc))

    raise ValueError(f"Cannot find {candidates}")


FEATURE_DIMENSION = {
    'weaver': {
        'wide_resnet28_2': 128,
        'wide_resnet28_10': 640,
        'preact_resnet18': 512,
    },
    'torchvision': {
        'vgg11': 4096,
        'vgg13': 4096,
        'vgg16': 4096,
        'vgg19': 4096,
        'vgg11_bn': 4096,
        'vgg13_bn': 4096,
        'vgg16_bn': 4096,
        'vgg19_bn': 4096,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560,
    },
}
