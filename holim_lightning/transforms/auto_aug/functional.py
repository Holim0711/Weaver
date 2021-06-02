from random import random
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw

__all__ = [
    'check_augment_min_max',
    'identity',
    'autocontrast',
    'equalize',
    'invert',
    'posterize',
    'solarize',
    'color',
    'contrast',
    'brightness',
    'sharpness',
    'rotate',
    'translateX',
    'translateY',
    'shearX',
    'shearY',
]

augment_bound = {
    'identity':      (0, 0),
    'autocontrast':  (0, 0),
    'equalize':      (0, 0),
    'invert':        (0, 0),
    'posterize':     (0, 8),
    'solarize':      (0, 256),
    'color':         (0.0, 1.0),
    'contrast':      (0.0, 1.0),
    'brightness':    (0.0, 1.0),
    'sharpness':     (0.0, 1.0),
    'rotate':        (0.0, 180.0),
    'translateX':    (0.0, 1.0),
    'translateY':    (0.0, 1.0),
    'shearX':        (0.0, 1.0),
    'shearY':        (0.0, 1.0),
}


def check_augment_min_max(augment_list):
    for op, min, max in augment_list:
        sub, sup = augment_bound[op]
        assert min >= sub, f"{op} min ({min} >= {sub})"
        assert max <= sup, f"{op} max ({max} <= {sup})"


def _random_flip(v):
    return v if random() < 0.5 else -v


def _affine(img, matrix, fillcolor):
    return img.transform(img.size, PIL.Image.AFFINE, matrix, fillcolor=fillcolor)


def identity(img, _, **kwargs):
    return img


def autocontrast(img, _, **kwargs):
    return PIL.ImageOps.autocontrast(img)


def equalize(img, _, **kwargs):
    return PIL.ImageOps.equalize(img)


def invert(img, _, **kwargs):
    return PIL.ImageOps.invert(img)


def posterize(img, v, **kwargs):
    return PIL.ImageOps.posterize(img, 8 - round(v))


def solarize(img, v, **kwargs):
    return PIL.ImageOps.solarize(img, 256 - round(v))


def color(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Color(img).enhance(1 + v)


def contrast(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v)


def brightness(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v)


def sharpness(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v)


def rotate(img, v, fillcolor='black'):
    v = _random_flip(v)
    return img.rotate(v, fillcolor=fillcolor)


def translateX(img, v, fillcolor='black'):
    v = _random_flip(v * img.size[0])
    return _affine(img, (1, 0, v, 0, 1, 0), fillcolor)


def translateY(img, v, fillcolor='black'):
    v = _random_flip(v * img.size[1])
    return _affine(img, (1, 0, 0, 0, 1, v), fillcolor)


def shearX(img, v, fillcolor='black'):
    v = _random_flip(v)
    return _affine(img, (1, v, 0, 0, 1, 0), fillcolor)


def shearY(img, v, fillcolor='black'):
    v = _random_flip(v)
    return _affine(img, (1, 0, 0, v, 1, 0), fillcolor)

