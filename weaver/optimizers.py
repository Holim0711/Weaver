import torch


def get_optim(module_or_params, name, **kwargs):
    if name == 'AdaBound':
        from adabound import AdaBound
        Optimizer = AdaBound
    elif name == 'LARS':
        from pl_bolts.optimizers import LARS
        Optimizer = LARS
    else:
        Optimizer = torch.optim.__dict__[name]

    if isinstance(module_or_params, torch.nn.Module):
        module_or_params = module_or_params.parameters()

    return Optimizer(module_or_params, **kwargs)


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
