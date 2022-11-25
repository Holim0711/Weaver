# Weaver PyTorch 🧶🧵

```python
from weaver import get_classifier, get_optimizer, get_scheduler, get_transforms
from torchvision.transforms import Compose
model = get_classifier('torchvision', 'resnet50')
optim = get_optimizer(model.parameters(), name='SGD', lr=1e-3)
sched = get_scheduler(optim, name='CosineAnnealingLR', T_max=10)
transform = Compose(get_transforms([
    {'name': 'RandAugment', 'num_ops': 2, 'magnitude': 10},
    {"name": "ToTensor"},
    {"name": "Normalize", "mean": "cifar10", "std": "cifar10"}
]))
```

## Install
```bash
pip install weaver-pytorch-rnx0dvmdxk
```

---------------------------------------

## API
### get_classifier(src, name, **kwargs)
- weaver: `wide_resnet{depth}_{width}`, `preact_resnet{depth}`
- torchvision: https://pytorch.org/vision/stable/models.html

### get_optimizer(params, name, **kwargs)
- PyTorch: https://pytorch.org/docs/stable/optim.html#algorithms
- AdaBelief: https://github.com/juntang-zhuang/Adabelief-Optimizer

### get_scheduler(optim, name, **kwargs)
- PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- Custom: `HalfCosineAnnealingLR`

### get_transform(name, **kwargs)
- PyTorch: https://pytorch.org/vision/stable/transforms.html
- Custom: `AllRandAugment`, `Cutout`, `Contain`

### get_transforms(kwargs_list)
- get list of transforms

### Others
- `weaver.optimizers.exclude_wd(module: Module, skip_list=['bias', 'bn'])`
- `weaver.optimizers.EMAModel(model: Module, alpha: float)`
- `weaver.datasets.IndexedDataset`
- `weaver.datasets.RandomSubset`
