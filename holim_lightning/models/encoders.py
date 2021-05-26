import torch
from .models import get_model


def get_encoder(*args, **kwargs):
    model = get_model(*args, **kwargs)
    if hasattr(model, 'fc'):
        dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    elif hasattr(model, '_fc'):
        dim = model._fc.in_features
        model._fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        # TODO: find first nn.Linear and check in_features
        raise NotImplementedError("Model not yet supported")
    else:
        raise NotImplementedError("Model not supported")
    return model, dim
