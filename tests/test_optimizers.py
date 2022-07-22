import unittest
from weaver.models import get_classifier
from weaver.optimizers import get_optim


class TestOptimizers(unittest.TestCase):

    def test_optimizers(self):
        model = get_classifier('torchvision', 'resnet18')

        optim = get_optim(model, 'SGD', lr=0.1)
        self.assertEqual(optim.__class__.__name__, 'SGD')

        optim = get_optim(model.parameters(), 'LARS', lr=0.1, momentum=0.9)
        self.assertEqual(optim.__class__.__name__, 'LARS')


if __name__ == "__main__":
    unittest.main()
