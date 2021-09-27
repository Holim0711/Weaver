# Weaver PyTorch 🧶🧵

Make hyper-parameters strings!

```
from weaver.models import get_model
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched

model = get_model('torchvision', 'resnet50', pretrained=False)
optim = get_optim(model, name='SGD', lr=1e-3)
sched = get_sched(optim, name='LinearWarmupCosineAnnealingLR', warmup_epochs=10, max_epochs=100)
```


## Models & Encoders
### Prototypes
- `get_model(src, name, pretrained=False, **kwargs)`
- `get_encoder(src, name, pretrained=False, **kwargs)`
- Encoders are just models whose last FCs are replaced by the identities!

### Model List
- `'weaver'`: `'wide_resnet28_{width}'`
- `'torchvision'`: https://pytorch.org/vision/stable/models.html
- `'lukemelas'`: https://github.com/lukemelas/EfficientNet-PyTorch


## Optimizers
### Prototypes
- `get_optim(module_or_params, name, **kwargs)`

### Optimizer List
- PyTorch
- LARS: https://github.com/PyTorchLightning/lightning-bolts


## Schedulers
### Prototypes
- `get_sched(optim, name, **kwargs)`

### Scheduler List
- PyTorch
- LinearWarmupCosineAnnealingLR: https://github.com/PyTorchLightning/lightning-bolts


## Transforms
### Prototypes
- `get_trfms(kwargs_list)`

### Transform List
- PyTorch
- AutoAugment, RandAugment, RandAugmentUDA
- Cutout, GaussianBlur, ContainResize
- EqTwinTransform, NqTwinTransform
