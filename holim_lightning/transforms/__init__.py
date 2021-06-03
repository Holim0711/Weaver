from .augments import AutoAugment, RandAugment
from .cutout import Cutout
from .contain_resize import ContainResize
from .gaussian_blur import GaussianBlur
from .twin_transforms import *
from torchvision import transforms as vtrfm


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


def get_Normalize(dataset, **kwargs):
    kwargs.update(rgb_stats[dataset])
    return vtrfm.Normalize(**kwargs)


def get_trfm(name, **kwargs):
    if name == 'AutoAugment':
        Transform = AutoAugment
    elif name == 'RandAugment':
        Transform = RandAugment
    elif name == 'ContainResize':
        Transform = ContainResize
    elif name == 'Cutout':
        Transform = Cutout
    elif name == 'GaussianBlur':
        Transform = GaussianBlur
    else:
        Transform = vtrfm.__dict__[name]

    if name == 'Normalize' and 'dataset' in kwargs:
        return get_Normalize(**kwargs)

    return Transform(**kwargs)


def get_trfms(kwargs_list):
    return vtrfm.Compose([get_trfm(**x) for x in kwargs_list])
