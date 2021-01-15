from torch.optim import lr_scheduler as torch_scheds


def get_sched(name):
    return torch_scheds.__dict__[name]
