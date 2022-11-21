from .custom import custom_transforms as ctrfm
from torchvision import transforms as vtrfm

__all__ = ['get_transform', 'get_transforms']

RGB = {
    "imagenet": {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    },
    "cifar10": {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2471, 0.2435, 0.2616),
    },
    "cifar100": {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'svhn': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    }
}


def get_transform_class(name: str) -> type:
    if cls := ctrfm.get(name):
        return cls
    if cls := vtrfm.__dict__.get(name):
        return cls
    raise ValueError()


def get_transform(name: str, **kwargs) -> callable:
    cls = get_transform_class(name)

    if isinstance(v := kwargs.get('fill'), str):
        kwargs['fill'] = tuple(round(x * 255) for x in RGB[v]['mean'])
    if isinstance(v := kwargs.get('mean'), str):
        kwargs['mean'] = RGB[v]['mean']
    if isinstance(v := kwargs.get('std'), str):
        kwargs['std'] = RGB[v]['std']
    if isinstance(v := kwargs.get('transforms'), list):
        kwargs['transforms'] = get_transforms(v)

    return cls(**kwargs)


def get_transforms(arglist: list[dict]) -> list[callable]:
    return [get_transform(**x) for x in arglist]
