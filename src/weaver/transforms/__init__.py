from torchvision import transforms
from .color import DATASET_RGB_STAT

__all__ = [
    'get_xform',
]


def get_xform_class(name):
    if name == 'AutoAugment':
        from .augments import AutoAugment
        return AutoAugment
    elif name == 'RandAugment':
        from .augments import RandAugment
        return RandAugment
    elif name == 'RandAugmentUDA':
        from .augments import RandAugmentUDA
        return RandAugmentUDA
    elif name == 'Cutout':
        from .cutout import Cutout
        return Cutout
    elif name == 'ContainResize':
        from .contain_resize import ContainResize
        return ContainResize
    elif name == 'GaussianBlur':
        from .gaussian_blur import GaussianBlur
        return GaussianBlur
    elif name == 'EqTwinTransform':
        from .twin_transforms import EqTwinTransform
        return EqTwinTransform
    elif name == 'NqTwinTransform':
        from .twin_transforms import NqTwinTransform
        return NqTwinTransform
    else:
        return transforms.__dict__[name]


def get_xform(name, **kwargs):
    Transform = get_xform_class(name)

    if name == 'Normalize' and 'dataset' in kwargs:
        kwargs.update(DATASET_RGB_STAT[kwargs.pop('dataset')])
    elif name == 'Compose':
        kwargs['transforms'] = [get_xform(**x) for x in kwargs['transforms']]
    elif name == 'RandomApply':
        kwargs['transforms'] = [get_xform(**x) for x in kwargs['transforms']]
    elif name == 'EqTwinTransform':
        kwargs['transforms'] = [get_xform(**x) for x in kwargs['transforms']]
    elif name == 'NqTwinTransform':
        kwargs['transforms1'] = [get_xform(**x) for x in kwargs['transforms1']]
        kwargs['transforms2'] = [get_xform(**x) for x in kwargs['transforms2']]

    return Transform(**kwargs)
