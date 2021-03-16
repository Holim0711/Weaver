import torch
import math


class CosineAnnealingWarmUp(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, T_max, T_warm=0, last_epoch=-1):
        def lr_lambda(t):
            if t < T_warm:
                return float(t) / float(T_warm)
            θ = float(t - T_warm) / float(T_max - T_warm)
            return 0.5 * (1 + math.cos(θ * math.pi))
        super().__init__(optimizer, lr_lambda, last_epoch)


def get_sched(optimizer, name, **kwargs):
    if name == 'CosineAnnealingWarmUp':
        return CosineAnnealingWarmUp(optimizer, **kwargs)
    return torch.optim.lr_scheduler.__dict__[name](optimizer, **kwargs)
