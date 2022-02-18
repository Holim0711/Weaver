import torch.nn as nn

__all__ = [
    'get_model',
    'get_vectorizer',
    'get_featurizer',
]


def get_model(src: str, name: str, **kwargs):
    src = src.lower()
    if src == 'weaver':
        from .custom import get_custom_model
        return get_custom_model(name, **kwargs)
    if src == 'torchssl':
        from .torchssl import get_torchssl_model
        return get_torchssl_model(name, **kwargs)
    if src == 'torchvision':
        import torchvision
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
        model.num_features = fc.in_features
        setattr(model, fc_name, nn.Identity())
    elif isinstance(fc, nn.Sequential):
        for i in range(len(fc) - 1, -1, -1):
            if isinstance(fc[i], nn.Linear):
                model.num_features = fc[i].in_features
                break
            if isinstance(fc[i], nn.Conv2d) and fc[i].kernel_size == (1, 1):
                model.num_features = fc[i].in_channels
                break
        else:
            raise Exception(f"There's no FC layer in {fc_name}")
        fc[i] = nn.Identity()
    else:
        raise NotImplementedError

    return model


def get_vectorizer(*args, **kwargs):
    model = get_model(*args, **kwargs)

    for fc_name in ['fc', '_fc', 'classifier']:
        if hasattr(model, fc_name):
            return change_fc(model, fc_name)

    raise ValueError("Cannot specify the name of the last fc layer")


def get_featurizer(*args, **kwargs):
    pass
