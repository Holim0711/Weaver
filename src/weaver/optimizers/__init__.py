import torch
from .custom import *

__all__ = ['get_optim']


def get_optim(module_or_params, name, **kwargs):
    Optimizer = LARS if name == 'LARS' else \
                torch.optim.__dict__[name]

    if isinstance(module_or_params, torch.nn.Module):
        module_or_params = module_or_params.parameters()

    return Optimizer(module_or_params, **kwargs)
