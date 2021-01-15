from torch import optim as torch_optims
from adabound import AdaBound


def get_optim(name):
    if name == 'AdaBound':
        return AdaBound
    return torch_optims.__dict__[name]
