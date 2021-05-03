import torch


class ConstantLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, lambda x: 1, last_epoch)


def get_sched(optim, name, **kwargs):
    if name is None or name == 'ConstantLR':
        Scheduler = ConstantLR
    elif name == 'LinearWarmupCosineAnnealingLR':
        from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
        Scheduler = LinearWarmupCosineAnnealingLR
    else:
        Scheduler = torch.optim.lr_scheduler.__dict__[name]

    return Scheduler(optim, **kwargs)
