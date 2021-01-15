import torch
from torchvision import models as torch_models
from efficientnet_pytorch import EfficientNet


def get_efficientnet(backbone, pretrained, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(backbone, **kwargs)
    else:
        return EfficientNet.from_name(backbone, **kwargs)


def get_torch_model(backbone, pretrained, **kwargs):
    return torch_models.__dict__[backbone](pretrained=pretrained, **kwargs)


def get_model(backbone, num_classes, pretrained=True):
    if backbone.startswith('efficientnet'):
        model = get_efficientnet(backbone, pretrained, num_classes=num_classes)
    else:
        model = get_torch_model(backbone, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_encoder(backbone, pretrained=True):
    if backbone.startswith('efficientnet'):
        model = get_efficientnet(backbone, pretrained, num_classes=num_classes)
        model._fc = torch.nn.Identity()
    else:
        model = get_torch_model(backbone, pretrained)
        model.fc = torch.nn.Identity()
    return model
