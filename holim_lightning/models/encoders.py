import torch
from .models import get_model


def get_encoder(**kwargs):
    model = get_model(**kwargs)
    if hasattr(model, 'fc'):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, '_fc'):
        dim = model._fc.in_features
        model._fc = torch.nn.Identity()
    else:
        raise NotImplementedError("This model doesn't have 'fc' or '_fc'")
    return model, dim
