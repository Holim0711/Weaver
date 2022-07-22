import math
from torch.optim.lr_scheduler import LambdaLR


class HalfCosineAnnealingLR(LambdaLR):

    def __init__(self, optimizer, T_max, eta_min=0,
                 last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min

        if eta_min != 0:
            raise NotImplementedError()

        def _lr_lambda(t):
            return max(0., math.cos(0.5 * math.pi * (t / T_max)))

        super().__init__(optimizer, _lr_lambda, last_epoch, verbose)
