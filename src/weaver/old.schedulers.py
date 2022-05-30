from math import cos, pi as π
from torch.optim.lr_scheduler import LambdaLR


__all__ = ['get_sched']


def get_sched(optim, name, **kwargs):
    return {
        'StepLR': StepLR,
        'MultiStepLR': MultiStepLR,
        'ExponentialLR': ExponentialLR,
        'CosineLR': CosineLR,
        'CosineAnnealingLR': CosineAnnealingLR,
    }[name](optim, **kwargs)


class StepLR(LambdaLR):
    def __init__(self, optimizer, T, γ=0.1, warmup=0, **kwargs):
        self.T = T
        self.γ = γ
        self.warmup = warmup

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return epoch / self.warmup
            return self.γ ** ((epoch - self.warmup) // self.T)

        super().__init__(optimizer, lr_lambda, **kwargs)

    def extend(self, m):
        self.T *= m
        self.warmup *= m


class MultiStepLR(LambdaLR):
    def __init__(self, optimizer, milestones, γ=0.1, warmup=0, **kwargs):
        self.milestones = milestones
        self.γ = γ
        self.warmup = warmup

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return epoch / self.warmup
            return self.γ ** sum(t <= epoch for t in self.milestones)

        super().__init__(optimizer, lr_lambda, **kwargs)

    def extend(self, m):
        self.milestones[:] = [t * m for t in self.milestones]
        self.warmup *= m


class ExponentialLR(LambdaLR):
    def __init__(self, optimizer, γ, warmup=0, **kwargs):
        self.γ = γ
        self.warmup = warmup

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return epoch / self.warmup
            return self.γ ** (epoch - self.warmup)

        super().__init__(optimizer, lr_lambda, **kwargs)

    def extend(self, m):
        self.γ **= 1 / m
        self.warmup *= m


class CosineLR(LambdaLR):
    def __init__(self, optimizer, T, ε=0, warmup=0, **kwargs):
        self.T = T
        self.ε = ε
        self.warmup = warmup

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return epoch / self.warmup
            return self.ε + (1 - self.ε) * cos((π / 2) * ((epoch - self.warmup) / (self.T - self.warmup)))

        super().__init__(optimizer, lr_lambda, **kwargs)

    def extend(self, m):
        self.T *= m
        self.warmup *= m


class CosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, T, ε=0, warmup=0, **kwargs):
        self.T = T
        self.ε = ε
        self.warmup = warmup

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return epoch / self.warmup
            return self.ε + (1 - self.ε) * (1 + cos(π * ((epoch - self.warmup) / (self.T - self.warmup)))) / 2

        super().__init__(optimizer, lr_lambda, **kwargs)

    def extend(self, m):
        self.T *= m
        self.warmup *= m
