import os
import sys
import json
import torch
from torchvision import datasets
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched
from holim_lightning.transforms import get_trfms


class Module(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(**self.hparams.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torch.nn.ModuleDict({
            '_train': torchmetrics.Accuracy(),
            '_val': torchmetrics.Accuracy(),
            '_test': torchmetrics.Accuracy(),
        })

    def shared_step(self, phase, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.accuracy['_' + phase].update(z.softmax(dim=-1), y)
        return {'loss': loss}

    def shared_epoch_end(self, phase, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.accuracy['_' + phase].compute()
        self.log_dict({
            f'{phase}/loss': loss,
            f'{phase}/acc': acc,
            'step': self.current_epoch,
        })
        self.accuracy['_' + phase].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step('train', batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end('train', outputs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('val', batch)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end('val', outputs)

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
    transform_train = get_trfms(hparams['transform']['train'])
    transform_valid = get_trfms(hparams['transform']['valid'])
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
            ModelCheckpoint(save_top_k=1, monitor='val/acc'),
            LearningRateMonitor(logging_interval=hparams['lr_dict']['interval']),
        ],
        **hparams['trainer'])

    trainer.fit(pl_module, dataloader_train, dataloader_valid)
    trainer.test(test_dataloaders=dataloader_valid)


if __name__ == "__main__":
    with open(sys.argv[1]) as file:
        hparams = json.load(file)
    run(hparams)
