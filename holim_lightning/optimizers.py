from torch import optim as torch_optims
from adabound import Adabound


def get_optim(name):
    if name == 'Adabound':
        return Adabound
    return torch_optims.__dict__[name]
