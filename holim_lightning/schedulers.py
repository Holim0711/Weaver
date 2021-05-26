def get_sched(optim, name, **kwargs):
    if name == 'LinearWarmupCosineAnnealingLR':
        from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
        Scheduler = LinearWarmupCosineAnnealingLR
    else:
        import torch
        Scheduler = torch.optim.lr_scheduler.__dict__[name]

    return Scheduler(optim, **kwargs)
