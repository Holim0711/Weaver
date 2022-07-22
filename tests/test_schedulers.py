import unittest
from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched


class TestSchedulers(unittest.TestCase):

    def test_optimizers(self):
        model = get_classifier('torchvision', 'resnet18')
        optim = get_optim(model, 'SGD', lr=1.0)
        sched = get_sched(optim, 'SequentialLR', schedulers=[
            {'name': 'LinearLR', 'start_factor': 0.2, 'total_iters': 4},
            {'name': 'CosineAnnealingLR', 'T_max': 5}
        ], milestones=[4])

        optim.step()

        result = []
        for _ in range(15):
            result.append(sched.get_last_lr()[0])
            sched.step()

        expect = [
            0.2, 0.4, 0.6, 0.8, 1.0,
            0.9045084971, 0.6545084971, 0.3454915028, 0.0954915028, 0.0,
            0.0954915028, 0.3454915028, 0.6545084971, 0.9045084971, 1.0,
        ]

        for a, b in zip(result, expect):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
