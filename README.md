# Weaver PyTorch 🧶🧵

Make hyper-parameters strings!

```
from weaver.models import get_model
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched

model = get_classifier('torchvision', 'resnet50', pretrained=False)
optim = get_optim(model, name='SGD', lr=1e-3)
sched = get_sched(optim, name='LinearWarmupCosineAnnealingLR', warmup_epochs=10, max_epochs=100)
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
- `get_classifier(src, name, pretrained=False, **kwargs)`
- `get_vectorizer(src, name, pretrained=False, **kwargs)`

### Classifier List
- `'weaver'`: `'wide_resnet{depth}_{width}'`, `'preact_resnet{depth}'`
- `'torchvision'`: https://pytorch.org/vision/stable/models.html


## Optimizers
### Prototypes
- `get_optim(module_or_params, name, **kwargs)`

### Optimizer List
- PyTorch
- LARS: https://github.com/PyTorchLightning/lightning-bolts


## Schedulers
### Prototypes
- `get_sched(optim, name, **kwargs)`

### Scheduler List (all the schedulers are custom!)
- `StepLR(optimizer, T, γ=0.1, warmup=0, **kwargs)`
- `MultiStepLR(optimizer, milestones, γ=0.1, warmup=0, **kwargs)`
- `ExponentialLR(optimizer, γ, warmup=0, **kwargs)`
- `CosineLR(optimizer, T, ε=0, warmup=0, **kwargs)`
- `CosineAnnealingLR(optimizer, T, ε=0, warmup=0, **kwargs)`

## Transforms
### Prototypes
- `get_trfms(kwargs_list)`

### Transform List
- PyTorch
- AutoAugment, RandAugment, RandAugmentUDA
- Cutout, GaussianBlur, ContainResize
- EqTwinTransform, NqTwinTransform
