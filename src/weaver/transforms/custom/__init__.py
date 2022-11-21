from .all_rand_augment import *
from .transforms import *

__all__ = ['custom_transforms']

custom_transforms = {
    'AllRandAugment': AllRandAugment,
    'Cutout': Cutout,
    'Contain': Contain,
}
