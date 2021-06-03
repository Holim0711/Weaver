__all__ = ['EqTwinTransform', 'NqTwinTransform']


class EqTwinTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class NqTwinTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return self.transform1(x), self.transform2(x)
