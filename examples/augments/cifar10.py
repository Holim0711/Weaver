import os
import json
import argparse

import torch
from torchvision.datasets import CIFAR10, CIFAR100
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_lr_dict
from holim_lightning.transforms import get_trfms


datasets = {
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
}


class Module(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(**self.hparams.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torch.nn.ModuleDict({
            'trn': torchmetrics.Accuracy(),
            'val': torchmetrics.Accuracy(),
        })

    def shared_step(self, phase, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.accuracy[phase].update(z.softmax(dim=-1), y)
        return {'loss': loss}

    def shared_epoch_end(self, phase, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'{phase}/loss', loss)
        acc = self.accuracy[phase].compute()
        self.log(f'{phase}/acc', acc)
        self.accuracy[phase].reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step('trn', batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end('trn', outputs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step('val', batch)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end('val', outputs)

    @property
    def num_devices(self) -> int:
        t = self.trainer
        return t.num_nodes * max(t.num_processes, t.num_gpus, t.tpu_cores or 0)

    @property
    def steps_per_epoch(self) -> int:
        num_accum = self.trainer.accumulate_grad_batches
        return len(self.train_dataloader()) // (num_accum * self.num_devices)

    def configure_optimizers(self):
        optim = get_optim(self, **self.hparams.optimizer)
        sched = get_lr_dict(optim, steps_per_epoch=self.steps_per_epoch, **self.hparams.lr_dict)
        return {'optimizer': optim, 'lr_scheduler': sched}


def run(hparams, args):
    callbacks = [ModelCheckpoint(save_top_k=0), LearningRateMonitor()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    num_devices = trainer.num_nodes * max(
        trainer.num_processes, trainer.num_gpus, trainer.tpu_cores or 0)

    Dataset = datasets[hparams['dataset']['name']]
    root = os.path.join('data', hparams['dataset']['name'])
    batch_size = hparams['dataset']['batch_size'] // num_devices

    transform = {
        'train': get_trfms(hparams['transform']['train']),
        'valid': get_trfms(hparams['transform']['valid']),
    }
    dataset = {
        'train': Dataset(root, train=True, transform=transform['train']),
        'valid': Dataset(root, train=False, transform=transform['valid']),
    }
    dataloader = {
        'train': torch.utils.data.DataLoader(
            dataset['train'], batch_size, shuffle=True,
            num_workers=os.cpu_count(), pin_memory=True),
        'valid': torch.utils.data.DataLoader(
            dataset['valid'], batch_size, shuffle=False,
            num_workers=os.cpu_count(), pin_memory=True),
    }

    model = Module(**hparams)
    trainer.fit(model, dataloader['train'], dataloader['valid'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    with open(args.config) as file:
        hparams = json.load(file)

    run(hparams, args)
