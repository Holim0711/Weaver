import torch
import math


class ConstantLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, lambda x: 1, last_epoch)


class CosineAnnealingWarmUp(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, T_max, T_warm=0, T_mute=0, last_epoch=-1):
        def lr_lambda(t):
            if t < T_mute:
                return 0
            elif t < T_warm:
                return (float(t) - float(T_mute)) / (float(T_warm) - float(T_mute))
            elif t < T_max:
                θ = float(t - T_warm) / float(T_max - T_warm)
                return 0.5 * (1 + math.cos(θ * math.pi))
            return 0.
        super().__init__(optimizer, lr_lambda, last_epoch)


def get_sched(optimizer, name, **kwargs):
    if name is None:
        return ConstantLR(optimizer, **kwargs)
    elif name == 'CosineAnnealingWarmUp':
        return CosineAnnealingWarmUp(optimizer, **kwargs)
    return torch.optim.lr_scheduler.__dict__[name](optimizer, **kwargs)
