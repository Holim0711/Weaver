from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim


def test_optimizers():
    model = get_model('torchvision', 'resnet18', pretrained=True)
    optim = get_optim(model.parameters(), 'SGD', lr=0.3)
    print(optim)
    optim = get_optim(model.parameters(), 'AdaBound', lr=1e-3, final_lr=0.1)
    print(optim)
    optim = get_optim(model.parameters(), 'LARS', lr=0.1, momentum=0.9, nesterov=True)
    print(optim)


if __name__ == "__main__":
    test_optimizers()
