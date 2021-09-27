from torchvision.transforms import Compose

__all__ = [
    'EqTwinTransform',
    'NqTwinTransform',
]


class EqTwinTransform:
    def __init__(self, transforms):
        self.t = Compose(transforms)

    def __call__(self, x):
        return self.t(x), self.t(x)


class NqTwinTransform:
    def __init__(self, transforms1, transforms2):
        self.t1 = Compose(transforms1)
        self.t2 = Compose(transforms2)

    def __call__(self, x):
        return self.t1(x), self.t2(x)
