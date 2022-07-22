from torch.optim import lr_scheduler as schedulers
from .half_cosine_annealing_lr import HalfCosineAnnealingLR

__all__ = ['get_sched']


def get_sched(optim, name, **kwargs):
    if 'schedulers' in kwargs:
        kwargs['schedulers'] = [
            get_sched(optim, **child_sched_kwargs)
            for child_sched_kwargs in kwargs['schedulers']
        ]

    if name == 'ChainedScheduler':
        return schedulers.ChainedScheduler(**kwargs)
    elif name == 'HalfCosineAnnealingLR':
        return HalfCosineAnnealingLR(optim, **kwargs)
    else:
        return schedulers.__dict__[name](optim, **kwargs)
