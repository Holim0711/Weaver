from typing import Iterable
from .functional import transform
import random

DEFAULT_POLICIES = {
    'ImageNet': [
        [["posterize", 0.4, 8], ["rotate", 0.6, 9]],
        [["solarize", 0.6, 5], ["autocontrast", 0.6, 5]],
        [["equalize", 0.8, 8], ["equalize", 0.6, 3]],
        [["posterize", 0.6, 7], ["posterize", 0.6, 6]],
        [["equalize", 0.4, 7], ["solarize", 0.2, 4]],

        [["equalize", 0.4, 4], ["rotate", 0.8, 8]],
        [["solarize", 0.6, 3], ["equalize", 0.6, 7]],
        [["posterize", 0.8, 5], ["equalize", 1.0, 2]],
        [["rotate", 0.2, 3], ["solarize", 0.6, 8]],
        [["equalize", 0.6, 8], ["posterize", 0.4, 6]],

        [["rotate", 0.8, 8], ["color", 0.4, 0]],
        [["rotate", 0.4, 9], ["equalize", 0.6, 2]],
        [["equalize", 0.0, 7], ["equalize", 0.8, 8]],
        [["invert", 0.6, 4], ["equalize", 1.0, 8]],
        [["color", 0.6, 4], ["contrast", 1.0, 8]],

        [["rotate", 0.8, 8], ["color", 1.0, 2]],
        [["color", 0.8, 8], ["solarize", 0.8, 7]],
        [["sharpness", 0.4, 7], ["invert", 0.6, 8]],
        [["shearX", 0.6, 5], ["equalize", 1.0, 9]],
        [["color", 0.4, 0], ["equalize", 0.6, 3]],

        [["equalize", 0.4, 7], ["solarize", 0.2, 4]],
        [["solarize", 0.6, 5], ["autocontrast", 0.6, 5]],
        [["invert", 0.6, 4], ["equalize", 1.0, 8]],
        [["color", 0.6, 4], ["contrast", 1.0, 8]],
        [["equalize", 0.8, 8], ["equalize", 0.6, 3]],
    ],
    'CIFAR10': [
        [["invert", 0.1, 7], ["contrast", 0.2, 6]],
        [["rotate", 0.7, 2], ["translateX", 0.3, 9]],
        [["sharpness", 0.8, 1], ["sharpness", 0.9, 3]],
        [["shearY", 0.5, 8],  ["translateY", 0.7, 9]],
        [["autocontrast", 0.5, 8], ["equalize", 0.9, 2]],

        [["shearY", 0.2, 7], ["posterize", 0.3, 7]],
        [["color", 0.4, 3], ["brightness", 0.6, 7]],
        [["sharpness", 0.3, 9], ["brightness", 0.7, 9]],
        [["equalize", 0.6, 5], ["equalize", 0.5, 1]],
        [["contrast", 0.6, 7], ["sharpness", 0.6, 5]],

        [["color", 0.7, 7], ["translateX", 0.5, 8]],
        [["equalize", 0.3, 7], ["autocontrast", 0.4, 8]],
        [["translateY", 0.4, 3], ["sharpness", 0.2, 6]],
        [["brightness", 0.9, 6], ["color", 0.2, 8]],
        [["solarize", 0.5, 2], ["invert", 0, 3]],

        [["equalize", 0.2, 0], ["autocontrast", 0.6, 0]],
        [["equalize", 0.2, 8], ["equalize", 0.6, 4]],
        [["color", 0.9, 9], ["equalize", 0.6, 6]],
        [["autocontrast", 0.8, 4], ["solarize", 0.2, 8]],
        [["brightness", 0.1, 3], ["color", 0.7, 0]],

        [["solarize", 0.4, 5], ["autocontrast", 0.9, 3]],
        [["translateY", 0.9, 9], ["translateY", 0.7, 9]],
        [["autocontrast", 0.9, 2], ["solarize", 0.8, 3]],
        [["equalize", 0.8, 8], ["invert", 0.1, 3]],
        [["translateY", 0.7, 9], ["autocontrast", 0.9, 1]],
    ],
    'SVHN': [
        [["shearX", 0.9, 4], ["invert", 0.2, 3]],
        [["shearY", 0.9, 8], ["invert", 0.7, 5]],
        [["equalize", 0.6, 5], ["solarize", 0.6, 6]],
        [["invert", 0.9, 3], ["equalize", 0.6, 3]],
        [["equalize", 0.6, 1], ["rotate", 0.9, 3]],

        [["shearX", 0.9, 4], ["autocontrast", 0.8, 3]],
        [["shearY", 0.9, 8], ["invert", 0.4, 5]],
        [["shearY", 0.9, 5], ["solarize", 0.2, 6]],
        [["invert", 0.9, 6], ["autocontrast", 0.8, 1]],
        [["equalize", 0.6, 3], ["rotate", 0.9, 3]],

        [["shearX", 0.9, 4], ["solarize", 0.3, 3]],
        [["shearY", 0.8, 8], ["invert", 0.7, 4]],
        [["equalize", 0.9, 5], ["translateY", 0.6, 6]],
        [["invert", 0.9, 4], ["equalize", 0.6, 7]],
        [["contrast", 0.3, 3], ["rotate", 0.8, 4]],

        [["invert", 0.8, 5], ["translateY", 0, 2]],
        [["shearY", 0.7, 6], ["solarize", 0.4, 8]],
        [["invert", 0.6, 4], ["rotate", 0.8, 4]],
        [["shearY", 0.3, 7], ["translateX", 0.9, 3]],
        [["shearX", 0.1, 6], ["invert", 0.6, 5]],

        [["solarize", 0.7, 2], ["translateY", 0.6, 7]],
        [["shearY", 0.8, 4], ["invert", 0.8, 8]],
        [["shearX", 0.7, 9], ["translateY", 0.8, 3]],
        [["shearY", 0.8, 5], ["autocontrast", 0.7, 3]],
        [["shearX", 0.7, 2], ["invert", 0.1, 5]],
    ],
}


class AutoAugment:

    AUGMENT_BOUND = {
        # AutoAugment: https://arxiv.org/pdf/1805.09501.pdf
        'identity':      (0, 0),
        'autocontrast':  (0, 0),
        'equalize':      (0, 0),
        'invert':        (0, 0),
        'posterize':     (4, 8),
        'solarize':      (0, 256),
        'color':         (0, 0.9),
        'contrast':      (0, 0.9),
        'brightness':    (0, 0.9),
        'sharpness':     (0, 0.9),
        'rotate':        (0, 30),
        'translateX':    (0, 0.453),  # (150/331)
        'translateY':    (0, 0.453),  # (150/331)
        'shearX':        (0, 0.3),
        'shearY':        (0, 0.3),
    }

    def __init__(self, policies='ImageNet', fillcolor=(128, 128, 128)):
        if isinstance(policies, str):
            policies = DEFAULT_POLICIES[policies]
        self.policies = policies
        self.fillcolor = fillcolor
        if isinstance(fillcolor, Iterable):
            self.fillcolor = tuple(fillcolor)

    def __call__(self, img):
        policy = random.choice(self.policies)
        for op, p, m in policy:
            if random.random() < p:
                min, max = self.AUGMENT_BOUND[op]
                v = (m / 10) * (max - min) + min
                img = transform(img, op, v, fillcolor=self.fillcolor)
        return img
