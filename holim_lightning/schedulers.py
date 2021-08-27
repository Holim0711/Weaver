def get_sched(optim, name, **kwargs):
    if name == 'LinearWarmupCosineAnnealingLR':
        from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
        Scheduler = LinearWarmupCosineAnnealingLR
    else:
        import torch
        Scheduler = torch.optim.lr_scheduler.__dict__[name]

    return Scheduler(optim, **kwargs)


def get_lr_dict(optim, scheduler, n_train_iters=None, **lr_dict):
    scheduler = dict(scheduler)
    if lr_dict['interval'] == 'step':
        if scheduler['name'] == 'CosineAnnealingLR':
            scheduler['T_max'] *= n_train_iters
        elif scheduler['name'] == 'CosineAnnealingWarmRestarts':
            scheduler['T_0'] *= n_train_iters
        elif scheduler['name'] == 'LinearWarmupCosineAnnealingLR':
            scheduler['warmup_epochs'] *= n_train_iters
            scheduler['max_epochs'] *= n_train_iters
        elif scheduler['name'] in {'CyclicLR', 'OneCycleLR'}:
            raise NotImplementedError("Unsupoorted lr scheduler yet")
    lr_dict['scheduler'] = get_sched(optim, **scheduler)
    return lr_dict
