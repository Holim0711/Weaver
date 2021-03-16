from torch.optim import lr_scheduler as torch_scheds
import math


class ConstantLR(torch_scheds.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, lambda x: 1, last_epoch)


class CosineAnnealingWarmUp(torch_scheds.LambdaLR):
    def __init__(self, optimizer, T_max, T_warm=0, last_epoch=-1):
        def lr_lambda(t):
            if t < T_warm:
                return float(t) / float(T_warm)
            θ = float(t - T_warm) / float(T_max - T_warm)
            return 0.5 * (1 + math.cos(θ * math.pi))
        super().__init__(optimizer, lr_lambda, last_epoch)


def get_sched(name):
    if name is None:
        return ConstantLR
    elif name == 'CosineAnnealingWarmUp':
        return CosineAnnealingWarmUp
    return torch_scheds.__dict__[name]
