import torch


def get_optim(param, name, **kwargs):
    if name == 'AdaBound':
        from adabound import AdaBound
        Optimizer = AdaBound
    elif name == 'LARS':
        from pl_bolts.optimizers import LARS
        Optimizer = LARS
    else:
        Optimizer = torch.optim.__dict__[name]

    return Optimizer(param, **kwargs)
