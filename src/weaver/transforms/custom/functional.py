from typing import Union
from PIL import Image, ImageDraw
from torch import rand
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

__all__ = ['contain', 'cutout']


def contain(
    img: Image.Image,
    size: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Union[str, list[float]] = 'black',
) -> Image.Image:
    x, y = img.size
    if x > y:
        x, y = size, size * y // x
    else:
        x, y = size * x // y, size
    img = resize(img, (x, y), interpolation=interpolation)
    new = Image.new('RGB', (size, size), color=fill)
    new.paste(img, ((size - x) // 2, (size - y) // 2))
    return new


def cutout(
    img: Image.Image,
    ratio: float,
    fill: Union[str, list[float]] = 'black',
) -> Image.Image:
    v = ratio / 2
    w, h = img.size
    x, y = rand(2)
    x0 = int(max(0, w * (x - v)))
    y0 = int(max(0, h * (y - v)))
    x1 = int(min(w, w * (x + v)))
    y1 = int(min(h, h * (y + v)))
    img = img.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), fill)
    return img
