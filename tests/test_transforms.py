import unittest
from weaver.transforms import get_xform


class TestTransforms(unittest.TestCase):

    def test_transforms(self):
        xform = get_xform(**{
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
        print(xform)


if __name__ == "__main__":
    unittest.main()
