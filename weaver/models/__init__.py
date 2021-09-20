import torch


def get_model(
    src: str,
    name: str,
    pretrained: bool = False,
    **kwargs
):
    if src == 'custom':
        from .custom import get_custom_model
        return get_custom_model(name, **kwargs)
    elif src == 'torchvision':
        from torchvision import models
        return models.__dict__[name](pretrained=pretrained, **kwargs)
    elif src == 'lukemelas':
        from efficientnet_pytorch import EfficientNet
        if pretrained:
            return EfficientNet.from_pretrained(name, **kwargs)
        else:
            return EfficientNet.from_name(name, **kwargs)
    else:
        raise ValueError(f"Unknown source: {src}")


def get_encoder(*args, **kwargs):
    model = get_model(*args, **kwargs)
    if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, '_fc') and isinstance(model._fc, torch.nn.Linear):
        dim = model._fc.in_features
        model._fc = torch.nn.Identity()
    else:
        raise NotImplementedError("Model not supported")
    model.num_features = dim
    return model
