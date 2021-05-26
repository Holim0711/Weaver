def get_model(src, name, pretrained=False, **kwargs):
    if src == 'torchvision':
        from torchvision import models
        return models.__dict__[name](pretrained=pretrained, **kwargs)
    elif src == 'lukemelas':
        from efficientnet_pytorch import EfficientNet
        if pretrained is True:
            return EfficientNet.from_pretrained(name, **kwargs)
        else:
            return EfficientNet.from_name(name, **kwargs)
    elif src == 'custom':
        from .custom import get_custom_model
        return get_custom_model(name, **kwargs)
    else:
        raise ValueError(f"Unknown model source {src}")
