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
    if src == 'cadene':
        import pretrainedmodels
        return pretrainedmodels.__dict__[name](**kwargs)
    if src == 'lukemelas':
        from efficientnet_pytorch import EfficientNet
        if kwargs.pop('pretrained', False):
            return EfficientNet.from_pretrained(name, **kwargs)
        else:
            return EfficientNet.from_name(name, **kwargs)

    raise ValueError(f"Unknown source: {src}")


def change_fc(model, fc_name):
    fc = getattr(model, fc_name)

    if isinstance(fc, nn.Linear):
        dim = fc.in_features
        setattr(model, fc_name, nn.Identity())
    else:
        raise NotImplementedError

    return model, dim


def get_vectorizer(*args, **kwargs):
    model = get_classifier(*args, **kwargs)

    for fc_name in ['fc', '_fc', 'classifier']:
        if hasattr(model, fc_name):
            return change_fc(model, fc_name)

    raise ValueError("Cannot specify the name of the last fc layer")
