# Weaver PyTorch 🧶🧵

Make hyper-parameters strings!

```
from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched
from weaver.transforms import get_xform

model = get_classifier('torchvision', 'resnet50')
optim = get_optim(model, name='SGD', lr=1e-3)
sched = get_sched(optim, name='CosineAnnealingLR', T_max=10)
xform = get_xform('Compose', transforms=[
    {'name': 'RandAugment', 'n': 2, 'm': 10},
    {"name": "ToTensor"},
    {"name": "Normalize", "dataset": "CIFAR10"}
])
```

## Installation
`pip install --index-url https://test.pypi.org/simple/ --no-deps weaver-pytorch-tools`

### dev
```
conda create --name weaver python=3
conda install pycodestyle
conda install pytorch torchvision ...  # https://pytorch.org/
pip install tensorboard  # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard
pip install -e .
```

## Models
### Prototypes
- `get_classifier(src, name, **kwargs)`
- `get_featurizer(src, name, **kwargs)`

### Classifier List
- `'weaver'`: `'wide_resnet{depth}_{width}'`, `'preact_resnet{depth}'`
- `'torchvision'`: https://pytorch.org/vision/stable/models.html


## Optimizers
### Prototypes
- `get_optim(module_or_params, name, **kwargs)`

### Optimizer List
- PyTorch: https://pytorch.org/docs/stable/optim.html#algorithms
- LARS: https://github.com/PyTorchLightning/lightning-bolts


## Schedulers
### Prototypes
- `get_sched(optim, name, **kwargs)`

### Scheduler List
- PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

## Transforms
### Prototypes
- `get_trfms(kwargs_list)`

### Transform List
- PyTorch: https://pytorch.org/vision/stable/transforms.html
- AutoAugment, RandAugment, RandAugmentUDA
- Cutout, GaussianBlur, ContainResize
- EqTwinTransform, NqTwinTransform
