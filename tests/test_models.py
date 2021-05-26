from holim_lightning.models import get_model, get_encoder


def test_torchvision():
    model = get_model('torchvision', 'resnet18', pretrained=True)
    print(model)
    encoder = get_encoder('torchvision', 'resnet18', pretrained=True)
    print(encoder)


def test_lukemelas():
    model = get_model('lukemelas', 'efficientnet-b0', pretrained=True, advprop=True)
    print(model)
    encoder = get_encoder('lukemelas', 'efficientnet-b0', pretrained=False)
    print(encoder)


def test_custom():
    model = get_model('custom', 'wide_resnet28_10', fixmatch=True)
    print(model)
    encoder = get_encoder('custom', 'wide_resnet28_10', fixmatch=True)
    print(encoder)


if __name__ == "__main__":
    test_torchvision()
    test_lukemelas()
    test_custom()
