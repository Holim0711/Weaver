from random import random
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw

__all__ = [
    'check_augment_min_max',
    'Identity',
    'AutoContrast',
    'Equalize',
    'Invert',
    'Posterize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'Rotate',
    'TranslateX',
    'TranslateY',
    'ShearX',
    'ShearY',
    'Cutout',
]

augment_bound = {
    'Identity':      (0, 0),
    'AutoContrast':  (0, 0),
    'Equalize':      (0, 0),
    'Invert':        (0, 0),
    'Posterize':     (0, 8),
    'Solarize':      (0, 256),
    'Color':         (0.0, 1.0),
    'Contrast':      (0.0, 1.0),
    'Brightness':    (0.0, 1.0),
    'Sharpness':     (0.0, 1.0),
    'Rotate':        (0.0, 180.0),
    'TranslateX':    (0.0, 1.0),
    'TranslateY':    (0.0, 1.0),
    'ShearX':        (0.0, 1.0),
    'ShearY':        (0.0, 1.0),
    'Cutout':        (0.0, 1.0),
}


def check_augment_min_max(augment_list):
    for op, min, max in augment_list:
        sub, sup = augment_bound[op]
        assert min >= sub, f"{op} min ({min} >= {sub})"
        assert max <= sup, f"{op} max ({max} <= {sup})"


def _random_flip(v):
    return v if random() < 0.5 else -v


def _affine(img, matrix, color):
    return img.transform(img.size, PIL.Image.AFFINE, matrix, fillcolor=color)


def Identity(img, _, **kwargs):
    return img


def AutoContrast(img, _, **kwargs):
    return PIL.ImageOps.autocontrast(img)


def Equalize(img, _, **kwargs):
    return PIL.ImageOps.equalize(img)


def Invert(img, _, **kwargs):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, **kwargs):
    return PIL.ImageOps.posterize(img, 8 - round(v))


def Solarize(img, v, **kwargs):
    return PIL.ImageOps.solarize(img, 256 - round(v))


def Color(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Color(img).enhance(1 + v)


def Contrast(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v)


def Brightness(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v)


def Sharpness(img, v, **kwargs):
    v = _random_flip(v)
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v)


def Rotate(img, v, color='black'):
    v = _random_flip(v)
    return img.rotate(v, fillcolor=color)


def TranslateX(img, v, color='black'):
    v = _random_flip(v * img.size[0])
    return _affine(img, (1, 0, v, 0, 1, 0), color)


def TranslateY(img, v, color='black'):
    v = _random_flip(v * img.size[1])
    return _affine(img, (1, 0, 0, 0, 1, v), color)


def ShearX(img, v, color='black'):
    v = _random_flip(v)
    return _affine(img, (1, v, 0, 0, 1, 0), color)


def ShearY(img, v, color='black'):
    v = _random_flip(v)
    return _affine(img, (1, 0, 0, v, 1, 0), color)


def Cutout(img, v, color='black'):
    w, h = img.size
    xc = random()
    yc = random()
    x0 = int(max(0, w * (xc - (v / 2))))
    y0 = int(max(0, h * (yc - (v / 2))))
    x1 = int(min(w, w * (xc + (v / 2))))
    y1 = int(min(h, h * (yc + (v / 2))))
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), color)
    return img
