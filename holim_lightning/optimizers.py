import torch
from pl_bolts.optimizers import LARS


def get_optim(params, name, **kwargs):
    if name == 'AdaBound':
        from adabound import AdaBound
        return AdaBound(params, **kwargs)
    elif name == 'LARS':
        return LARS(params, **kwargs)
    return torch.optim.__dict__[name](params, **kwargs)
