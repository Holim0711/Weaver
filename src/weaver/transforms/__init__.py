from torchvision import transforms

__all__ = [
    'get_trfms',
]

rgb_stats = {
    "ImageNet": {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    },
    "CIFAR10": {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616),
    },
    "CIFAR100": {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'SVHN': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    }
}


def get_trfm_class(name):
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


def get_trfm(name, **kwargs):
    Transform = get_trfm_class(name)

    if name == 'Normalize' and 'dataset' in kwargs:
        kwargs.update(rgb_stats[kwargs.pop('dataset')])
    elif name == 'RandomApply':
        kwargs['transforms'] = [get_trfm(**x) for x in kwargs['transforms']]
    elif name == 'EqTwinTransform':
        kwargs['transforms'] = [get_trfm(**x) for x in kwargs['transforms']]
    elif name == 'NqTwinTransform':
        kwargs['transforms1'] = [get_trfm(**x) for x in kwargs['transforms1']]
        kwargs['transforms2'] = [get_trfm(**x) for x in kwargs['transforms2']]

    return Transform(**kwargs)


def get_trfms(kwargs_list):
    return transforms.Compose([get_trfm(**x) for x in kwargs_list])
