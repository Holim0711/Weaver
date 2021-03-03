from torch import optim as torch_optims


def get_optim(name):
    return torch_optims.__dict__[name]
