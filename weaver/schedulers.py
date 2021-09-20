import torch


def get_scheduler(name):
    if name == 'LinearWarmupCosineAnnealingLR':
        from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
        return LinearWarmupCosineAnnealingLR
    else:
        return torch.optim.lr_scheduler.__dict__[name]


def get_sched(optim, name, **kwargs):
    Scheduler = get_scheduler(name)
    return Scheduler(optim, **kwargs)


epoch_fields = {
    'CosineAnnealingLR': ['T_max'],
    'CosineAnnealingWarmRestarts': ['T_0'],
    'LinearWarmupCosineAnnealingLR': ['warmup_epochs', 'max_epochs'],
}


def get_lr_dict(optim, scheduler, steps_per_epoch=None, **lr_dict):
    scheduler = dict(scheduler)

    if scheduler['name'] in {'CyclicLR', 'OneCycleLR'}:
        raise NotImplementedError("Unsupoorted scheduler yet")

    if lr_dict.get('interval') == 'step':
        if scheduler['name'] in epoch_fields:
            for k in epoch_fields[scheduler['name']]:
                scheduler[k] *= steps_per_epoch

    lr_dict['scheduler'] = get_sched(optim, **scheduler)
    return lr_dict
