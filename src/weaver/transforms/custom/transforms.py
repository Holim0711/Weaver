from typing import Union
from PIL.Image import Image
from torchvision.transforms import InterpolationMode
from .functional import *

__all__ = ['Contain', 'Cutout']


class Contain:
    def __init__(
        self,
        size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Union[str, list[float]] = 'black',
    ):
        self.size = size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img: Image) -> Image:
        return contain(img, self.size, self.interpolation, self.fill)

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(size={self.size}, '
            f'interpolation={self.interpolation.value}, '
            f'fill={self.fill})'
        )


class Cutout:
    def __init__(self, ratio: float, fill: Union[str, list[float]] = 'black'):
        self.ratio = ratio
        self.fill = fill

    def __call__(self, img: Image) -> Image:
        return cutout(img, self.ratio, self.fill)

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(ratio={self.ratio}, '
            f'fill={self.fill})'
        )
