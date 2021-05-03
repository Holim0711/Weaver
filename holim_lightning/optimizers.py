import torch


def get_optim(params, name, **kwargs):
    if name == 'AdaBound':
        from adabound import AdaBound
        return AdaBound(params, **kwargs)
    elif name == 'LARS':
        from pl_bolts.optimizers import LARS
        return LARS(params, **kwargs)
    return torch.optim.__dict__[name](params, **kwargs)
