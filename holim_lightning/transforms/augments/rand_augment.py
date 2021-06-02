from typing import Iterable
from . import functional as f
import random

__all__ = [
    'RandAugment',
    'RandAugmentUDA',
]


class RandAugment:
    QUANTIZE_LEVEL = 10

    DEFAULT_AUGMENT_LIST = [
        # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
        # AutoAugment: https://arxiv.org/pdf/1805.09501.pdf
        ('identity',      0, 0),
        ('autocontrast',  0, 0),
        ('equalize',      0, 0),
        ('posterize',     4, 8),
        ('solarize',      0, 256),
        ('color',         0, 0.9),
        ('contrast',      0, 0.9),
        ('brightness',    0, 0.9),
        ('sharpness',     0, 0.9),
        ('rotate',        0, 30),
        ('translateX',    0, 0.453),  # (150/331)
        ('translateY',    0, 0.453),  # (150/331)
        ('shearX',        0, 0.3),
        ('shearY',        0, 0.3),
    ]

    def __init__(self, n, m, augment_list=None, fillcolor='black'):
        self.n = int(n)
        self.m = int(m)
        self.augment_list = augment_list
        self.fillcolor = fillcolor
        if augment_list is None:
            self.augment_list = self.DEFAULT_AUGMENT_LIST
        if isinstance(fillcolor, Iterable):
            self.fillcolor = tuple(fillcolor)
        f.check_augment_min_max(self.augment_list)

    def transform(self, img, op, v):
        return f.__dict__[op](img, v, fillcolor=self.fillcolor)

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
            if random.random() < 0.5:
                v = random.random() * (max - min) + min
                img = self.transform(img, op, v)
        return img
