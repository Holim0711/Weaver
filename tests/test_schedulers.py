from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


def test_schedulers():
    model = get_model('torchvision', 'resnet18', pretrained=True)
    optim = get_optim(model.parameters(), 'SGD', lr=0.1)
    sched = get_sched(optim, 'StepLR', step_size=5, gamma=0.1)
    print(sched)
    sched = get_sched(optim, 'LinearWarmupCosineAnnealingLR', warmup_epochs=10, max_epochs=100)
    print(sched)


if __name__ == "__main__":
    test_schedulers()
