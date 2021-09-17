# 🧦 MEIAS PyTorch

Make Everything In A String!

```
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched

model = get_model('torchvision', 'resnet50', pretrained=False)
optim = get_optim(model.parameters(), name='SGD', lr=1e-3)
sched = get_sched(optim, name='LinearWarmupCosineAnnealingLR', warmup_epochs=10, max_epochs=100)
```


## Models & Encoders
### Prototypes
- `get_model(src, name, pretrained=False, **kwargs)`
- `get_encoder(src, name, pretrained=False, **kwargs)`
- Encoders are models without the last fully-connected layer

### Usage
```
from holim_lightning.models import get_model
model = get_model('torchvision', 'resnet50', pretrained=False)
model = get_model('lukemelas', 'efficientnet-b0', pretrained=True)
model = get_model('custom', 'wide_resnet28_2')  # all custom models are not pretrained.
```

### Source List
- torchvision: https://pytorch.org/vision/stable/models.html
- lukemelas: https://github.com/lukemelas/EfficientNet-PyTorch
- custom: This Repository!

### Custom Model List
- `'wide_resnet28_{width}'`

## Optimizers
### Usage

Prototype: `get_optim(param, name, **kwargs)`

```
from holim_lightning.optimizers import get_optim
optim = get_optim(model.parameters(), name='SGD', lr=1e-3)
```

### Source List
- PyTorch
- LARS: https://github.com/PyTorchLightning/lightning-bolts
- AdaBound: https://github.com/Luolc/AdaBound

## Schedulers
### Usage

Prototype: `get_sched(optim, name, **kwargs)`

```
from holim_lightning.schedulers import get_sched
sched = get_sched(optim, name='LinearWarmupCosineAnnealingLR', warmup_epochs=10, max_epochs=100)
```

### Source List
- PyTorch
- LinearWarmupCosineAnnealingLR: https://github.com/PyTorchLightning/lightning-bolts
