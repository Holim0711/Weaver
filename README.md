# Weaver PyTorch 🧶🧵

Make hyper-parameters strings!

```
from weaver import get_classifier, get_optimizer, get_scheduler, get_transforms
from torchvision.transforms import Compose
model = get_classifier('torchvision', 'resnet50')
optim = get_optimizer(model.parameters(), name='SGD', lr=1e-3)
sched = get_scheduler(optimizer, name='CosineAnnealingLR', T_max=10)
transform = Compose(get_transforms([
    {'name': 'RandAugment', 'num_ops': 2, 'magnitude': 10},
    {"name": "ToTensor"},
    {"name": "Normalize", "mean": "cifar10", "std": "cifar10"}
]))
```

## Installation
`pip install .`

## Models
### Prototypes
- `get_classifier(src: str, name: str, **kwargs)`

### Classifier `(src, name)` List
- `'weaver'`: `'wide_resnet{depth}_{width}'`, `'preact_resnet{depth}'`
- `'torchvision'`: https://pytorch.org/vision/stable/models.html


## Optimizers
### Prototypes
- `get_optimizer(params: list, name: str, **kwargs)`
- `exclude_wd(module: torch.nn.Module, skip_list=['bias', 'bn'])`
- `EMAModel(model: torch.nn.Module, alpha: float)`

### Optimizer List
- PyTorch: https://pytorch.org/docs/stable/optim.html#algorithms
- AdaBelief: https://github.com/juntang-zhuang/Adabelief-Optimizer


## Schedulers
### Prototypes
- `get_scheduler(optim: Optimizer, name: str, **kwargs)`

### Scheduler List
- PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

## Transforms
### Prototypes
- `get_transform(name: str, **kwargs)`
- `get_transforms(kwargs_list: list)`

### Transform List
- PyTorch: https://pytorch.org/vision/stable/transforms.html
- Custom: `AllRandAugment`, `Cutout`, `Contain`
