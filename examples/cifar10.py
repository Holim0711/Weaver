import os
import torch
from torchvision import transforms, datasets
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched
from holim_lightning.transforms import RandAugment


class Module(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(**self.hparams.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torch.nn.ModuleDict({
            'training': torchmetrics.Accuracy(),
            'validation': torchmetrics.Accuracy(),
            'test': torchmetrics.Accuracy(),
        })

    def shared_step(self, phase, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.accuracy[phase].update(z.softmax(dim=-1), y)
        return {'loss': loss}

    def shared_epoch_end(self, phase, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.accuracy[phase].compute()
        self.log_dict({
            f'{phase}/loss': loss,
            f'{phase}/acc': acc,
            'step': self.current_epoch,
        })
        self.accuracy[phase].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step('training', batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end('training', outputs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('validataion', batch)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end('validataion', outputs)

    def test_step(self, batch, batch_idx):
        return self.shared_step('test', batch)

    def test_epoch_end(self, outputs):
        self.shared_epoch_end('test', outputs)

    def configure_optimizers(self):
        optim = get_optim(self, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'name': "lr",
                'scheduler': sched,
                **self.hparams.lr_dict
            },
        }


def run(hparams):
    transform_train = transforms.Compose([
        RandAugment(**hparams['rand_augment'], color=[125, 123, 114]),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2471, 0.2435, 0.2616]),
    ])
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2471, 0.2435, 0.2616]),
    ])
    dataset_train = datasets.CIFAR10(
        './data', train=True, transform=transform_train, download=True)
    dataset_valid = datasets.CIFAR10(
        './data', train=False, transform=transform_valid, download=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=hparams['dataset']['batch_size'],
        shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=hparams['dataset']['batch_size'],
        shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    pl_module = Module(**hparams)

    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='validataion/acc'),
            LearningRateMonitor(logging_interval=hparams['lr_dict']['interval']),
        ],
        **hparams['trainer'])

    trainer.fit(pl_module, dataloader_train, dataloader_valid)
    trainer.test(test_dataloaders=dataloader_valid)


if __name__ == "__main__":
    run({
        'rand_augment': {
            'n': 3,
            'm': 5
        },
        'dataset': {
            'batch_size': 128,
        },
        'model': {
            'src': 'custom',
            'name': 'wide_resnet28_10',
            'num_classes': 10
        },
        'optimizer': {
            'name': 'SGD',
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'nesterov': True
        },
        'scheduler': {
            'name': 'LinearWarmupCosineAnnealingLR',
            'warmup_epochs': 5 * 391,
            'max_epochs': 200 * 391
        },
        'lr_dict': {
            'interval': 'step',
            'frequency': 1
        },
        'trainer': {
            'gpus': 1,
            'max_epochs': 200
        }
    })
