from torch import optim as torch_optims


def get_optim(name):
    if name == 'AdaBound':
        from adabound import AdaBound
        return AdaBound
    return torch_optims.__dict__[name]
