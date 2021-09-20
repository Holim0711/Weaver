import torch.nn as nn


def get_model(
    src: str,
    name: str,
    pretrained: bool = False,
    **kwargs
):
    if src == 'custom':
        from .custom import get_custom_model
        return get_custom_model(name, **kwargs)

    if src == 'torchvision':
        from torchvision import models
        return models.__dict__[name](pretrained=pretrained, **kwargs)

    if src == 'lukemelas':
        from efficientnet_pytorch import EfficientNet
        if pretrained:
            return EfficientNet.from_pretrained(name, **kwargs)
        else:
            return EfficientNet.from_name(name, **kwargs)

    raise ValueError(f"Unknown source: {src}")


def change_last_fc(model, last_fc_name):
    fc = getattr(model, last_fc_name)

    if isinstance(fc, nn.Linear):
        model.num_features = fc.in_features
        setattr(model, last_fc_name, nn.Identity())
    elif isinstance(fc, nn.Sequential) and isinstance(fc[-1], nn.Linear):
        model.num_features = fc[-1].in_features
        fc[-1] = nn.Identity()
    else:
        # torchvision SqueezeNet: classifier[-1] == AvgPool...
        raise NotImplementedError("Cannot specify where the last fc is")

    return model


def get_encoder(*args, **kwargs):
    model = get_model(*args, **kwargs)

    for last_fc_name in ['fc', '_fc', 'classifier']:
        if hasattr(model, last_fc_name):
            return change_last_fc(model, last_fc_name)

    raise ValueError("Cannot specify the name of the last fc layer")
