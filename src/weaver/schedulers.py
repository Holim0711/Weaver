from torch.optim import lr_scheduler as schedulers

__all__ = ['get_sched']


def get_sched(optim, name, **kwargs):
    if name in {'ChainedScheduler', 'SequentialLR'}:
        kwargs['schedulers'] = [get_sched(**x) for x in kwargs['schedulers']]
    return schedulers.__dict__[name](optim, **kwargs)
