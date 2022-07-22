import unittest
from weaver.transforms import get_transform


class TestTransforms(unittest.TestCase):

    def test_transforms(self):
        xfm = get_transform(**{
            'name': 'NqTwinTransform',
            'transforms1': [
                {'name': 'RandAugment', 'n': 2, 'm': 10},
                {"name": "Cutout", "ratio": 0.5, "fillcolor": "CIFAR10"},
                {"name": "RandomCrop", "size": 32, "padding": 4},
                {"name": "RandomHorizontalFlip"},
                {"name": "ToTensor"},
                {"name": "Normalize", "dataset": "CIFAR10"}
            ],
            'transforms2': [
                {'name': 'AutoAugment'},
                {"name": "Cutout", "ratio": 0.5, "fillcolor": "black"},
                {"name": "RandomCrop", "size": 32, "padding": 4},
                {"name": "RandomHorizontalFlip"},
                {"name": "ToTensor"},
                {"name": "Normalize", "dataset": "ImageNet"}
            ]
        })
        print(xfm)


if __name__ == "__main__":
    unittest.main()
