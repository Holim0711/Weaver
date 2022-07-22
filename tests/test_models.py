import unittest
import torch
from torch.utils.tensorboard import SummaryWriter
from weaver.models import get_classifier, get_vectorizer


class TestModels(unittest.TestCase):

    def visualize_models(self, src, name, input_shape, **kwargs):
        writer = SummaryWriter(f'runs/{src}/{name}')

        cls = get_classifier(src, name, **kwargs)
        vec = get_vectorizer(src, name, **kwargs)

        x = torch.randn(input_shape)

        writer.add_graph(cls, x)
        writer.add_graph(vec, x)
        writer.close()

    def test_torchvision_models(self):
        src = 'torchvision'
        models = ['resnet18', 'efficientnet_b0']
        input_shape = (1, 3, 224, 224)
        for model in models:
            self.visualize_models(src, model, input_shape)

    def test_weaver_models(self):
        src = 'weaver'
        models = ['wide_resnet28_2', 'preact_resnet18']
        input_shape = (1, 3, 32, 32)
        for model in models:
            self.visualize_models(src, model, input_shape)


if __name__ == "__main__":
    unittest.main()
