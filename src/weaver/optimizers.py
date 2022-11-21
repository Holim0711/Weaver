import torch

__all__ = ['get_optimizer', 'exclude_wd', 'EMAModel']


def get_optimizer(params, name, **kwargs):
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


class EMAModel(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model: torch.nn.Module, alpha: float):
        super().__init__(model, avg_fn=lambda m, x, _: alpha*m + (1-alpha)*x)

    def update_parameters(self, model):
        super().update_parameters(model)
        # BatchNorm buffers are already EMA
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))
