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

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.t.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


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

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    -- transform1 --'
        for t in self.t1.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n    -- transform2 --'
        for t in self.t2.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
