import torch

__all__ = ['get_optim', 'exclude_wd']


def get_optim(params, name, **kwargs):
    if name == 'AdaBelief':
        from adabelief_pytorch import AdaBelief
        Optimizer = AdaBelief
    else:
        Optimizer = torch.optim.__dict__[name]

    return Optimizer(params, **kwargs)


def exclude_wd(module, skip_list=['bias', 'bn']):
    params = []
    excluded_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        elif any(k in name for k in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return [{'params': params}, {'params': excluded_params, 'weight_decay': 0}]
