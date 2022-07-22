import unittest
import torch
from torch.utils.tensorboard import SummaryWriter
from weaver.models import get_classifier, get_featurizer, FEATURE_DIMENSION


class TestModels(unittest.TestCase):

    def template_classifier_tests(self, src, names, input_shape, **kwargs):
        for name in names:
            m = get_classifier(src, name, **kwargs)
            with SummaryWriter(f'runs/{src}/{name}') as writer:
                writer.add_graph(m, torch.randn(input_shape))

    def template_featurizer_tests(self, src, names, input_shape, **kwargs):
        for name in names:
            m = get_featurizer(src, name, **kwargs)
            m.eval()
            with torch.no_grad():
                v = m(torch.randn(input_shape))
            self.assertEqual(v.shape[0], input_shape[0])
            self.assertEqual(v.shape[1], FEATURE_DIMENSION[src][name])

    def test_torchvision_classifiers(self):
        self.template_classifier_tests(
            'torchvision',
            ['resnet18', 'efficientnet_b0'],
            (1, 3, 224, 224)
        )

    def test_weaver_classifiers(self):
        self.template_classifier_tests(
            'weaver',
            ['wide_resnet28_2', 'preact_resnet18'],
            (1, 3, 32, 32)
        )

    def test_torchvision_featurizers(self):
        self.template_featurizer_tests(
            'torchvision',
            ['resnet18', 'efficientnet_b0'],
            (1, 3, 224, 224)
        )

    def test_weaver_featurizers(self):
        self.template_featurizer_tests(
            'weaver',
            ['wide_resnet28_2', 'preact_resnet18'],
            (1, 3, 32, 32)
        )


if __name__ == "__main__":
    unittest.main()
