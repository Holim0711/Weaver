from . import functional as f
import random

__all__ = [
    'RandAugment',
    'RandAugmentUDA',
]


class RandAugment:
    QUANTIZE_LEVEL = 10

    DEFAULT_AUGMENT_LIST = [
        ('Identity',      0, 0),
        ('AutoContrast',  0, 0),
        ('Equalize',      0, 0),
        ('Invert',        0, 0),
        ('Posterize',     0, 4),
        ('Solarize',      0, 256),
        ('Color',         0, 0.9),
        ('Contrast',      0, 0.9),
        ('Brightness',    0, 0.9),
        ('Sharpness',     0, 0.9),
        ('Rotate',        0, 30),
        ('TranslateX',    0, 0.45),
        ('TranslateY',    0, 0.45),
        ('ShearX',        0, 0.3),
        ('ShearY',        0, 0.3),
        ('Cutout',        0, 0.2),
    ]

    def __init__(self, n, m, augment_list=None, color='black'):
        self.n = int(n)
        self.m = int(m)
        if augment_list is None:
            augment_list = self.DEFAULT_AUGMENT_LIST
        f.check_augment_min_max(augment_list)
        self.augment_list = list(augment_list)
        self.color = color

    def transform(self, img, op, v):
        return f.__dict__[op](img, v, color=self.color)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min, max in ops:
            v = self.m / self.QUANTIZE_LEVEL
            v = v * (max - min) + min
            img = self.transform(img, op, v)
        return img


class RandAugmentUDA(RandAugment):
    DEFAULT_AUGMENT_LIST = [
        ('Identity',      0, 0),
        ('AutoContrast',  0, 0),
        ('Equalize',      0, 0),
        ('Posterize',     0, 4),
        ('Solarize',      0, 256),
        ('Color',         0.05, 0.95),
        ('Contrast',      0.05, 0.95),
        ('Brightness',    0.05, 0.95),
        ('Sharpness',     0.05, 0.95),
        ('Rotate',        0, 30),
        ('TranslateX',    0, 0.3),
        ('TranslateY',    0, 0.3),
        ('ShearX',        0, 0.3),
        ('ShearY',        0, 0.3),
    ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min, max in ops:
            v = random.randint(1, self.m) / self.QUANTIZE_LEVEL
            v = v * (max - min) + min
            if random.random() < 0.5:
                img = self.transform(img, op, v)
        return img
