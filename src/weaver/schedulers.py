import math
from torch.optim import lr_scheduler

__all__ = ['get_scheduler']


class HalfCosineAnnealingLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, T_max, last_epoch=-1, verbose=False):
        def _lr_lambda(t):
            return max(0., math.cos(0.5 * math.pi * (t / T_max)))
        super().__init__(optimizer, _lr_lambda, last_epoch, verbose)


def get_scheduler(optim, name, **kwargs):
    if 'schedulers' in kwargs:
        kwargs['schedulers'] = [
            get_scheduler(optim, **child_sched_kwargs)
            for child_sched_kwargs in kwargs['schedulers']
        ]

    if name == 'ChainedScheduler':
        return lr_scheduler.ChainedScheduler(**kwargs)
    elif name == 'HalfCosineAnnealingLR':
        return HalfCosineAnnealingLR(optim, **kwargs)
    else:
        return lr_scheduler.__dict__[name](optim, **kwargs)
