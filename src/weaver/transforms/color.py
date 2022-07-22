from PIL.ImageColor import getrgb as pil_getrgb

DATASET_RGB_STAT = {
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


def getrgb(color):
    if color in DATASET_RGB_STAT:
        rgb = DATASET_RGB_STAT[color]['mean']
        return tuple(round(v * 255) for v in rgb)
    else:
        return pil_getrgb(color)
