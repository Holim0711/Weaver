from typing import Iterable
from .functional import transform, check_augment_min_max
from ..color import getrgb
import random

__all__ = [
    'RandAugment',
    'RandAugmentUDA',
]


class RandAugment:
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

    def __init__(self, n, m, augment_list=None, fillcolor=(128, 128, 128)):
        self.n = int(n)
        self.m = int(m)
        self.augment_list = augment_list
        self.fillcolor = fillcolor
        if augment_list is None:
            self.augment_list = self.DEFAULT_AUGMENT_LIST
        if isinstance(fillcolor, Iterable):
            self.fillcolor = tuple(fillcolor)
        elif isinstance(fillcolor, str):
            self.fillcolor = getrgb(fillcolor)
        check_augment_min_max(self.augment_list)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min, max in ops:
            v = self.m / 10
            v = v * (max - min) + min
            img = transform(img, op, v, fillcolor=self.fillcolor)
        return img

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(n={self.n}, m={self.m}, fillcolor={self.fillcolor})'
        )


class RandAugmentUDA:
    DEFAULT_AUGMENT_LIST = [
        ('identity',      0, 0),
        ('autocontrast',  0, 0),
        ('equalize',      0, 0),
        ('posterize',     4, 8),
        ('solarize',      0, 256),
        ('color',         0.05, 0.95),
        ('contrast',      0.05, 0.95),
        ('brightness',    0.05, 0.95),
        ('sharpness',     0.05, 0.95),
        ('rotate',        0, 30),
        ('translateX',    0, 0.3),
        ('translateY',    0, 0.3),
        ('shearX',        0, 0.3),
        ('shearY',        0, 0.3),
    ]

    def __init__(self, n, augment_list=None, fillcolor=(128, 128, 128)):
        self.n = int(n)
        self.augment_list = augment_list
        self.fillcolor = fillcolor
        if augment_list is None:
            self.augment_list = self.DEFAULT_AUGMENT_LIST
        if isinstance(fillcolor, str):
            self.fillcolor = getrgb(fillcolor)
        elif isinstance(fillcolor, Iterable):
            self.fillcolor = tuple(fillcolor)
        check_augment_min_max(self.augment_list)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min, max in ops:
            if random.random() < 0.5:
                v = random.random() * (max - min) + min
                img = transform(img, op, v, fillcolor=self.fillcolor)
        return img

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(n={self.n}, fillcolor={self.fillcolor})'
        )
