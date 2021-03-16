import torch


def get_optim(params, name, **kwargs):
    if name == 'AdaBound':
        from adabound import AdaBound
        return AdaBound(params, **kwargs)
    return torch.optim.__dict__[name](params, **kwargs)
