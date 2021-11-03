from torchvision.transforms import Compose

__all__ = [
    'EqTwinTransform',
    'NqTwinTransform',
]


class EqTwinTransform:
    def __init__(self, transforms):
        self.t = transforms
        if not isinstance(self.t, Compose):
            self.t = Compose(self.t)

    def __call__(self, x):
        return self.t(x), self.t(x)


class NqTwinTransform:
    def __init__(self, transforms1, transforms2):
        self.t1 = transforms1
        self.t2 = transforms2
        if not isinstance(self.t1, Compose):
            self.t1 = Compose(self.t1)
        if not isinstance(self.t2, Compose):
            self.t2 = Compose(self.t2)

    def __call__(self, x):
        return self.t1(x), self.t2(x)
