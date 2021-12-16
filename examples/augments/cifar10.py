import os
import json
import argparse

import torch
from torchvision.datasets import CIFAR10, CIFAR100
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from weaver.models import get_model
from weaver.optimizers import get_optim
from weaver.schedulers import get_sched
from weaver.transforms import get_trfms


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

    def configure_optimizers(self):
        optim = get_optim(self, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        sched.extend(self.hparams.steps_per_epoch)
        return {'optimizer': optim,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}


def run(hparams, args):
    hparams['random_seed'] = args.random_seed
    seed_everything(args.random_seed)

    callbacks = [ModelCheckpoint(save_top_k=0), LearningRateMonitor()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    num_devices = trainer.num_nodes * max(
        trainer.num_processes, trainer.num_gpus, trainer.tpu_cores or 0)
    num_accum = trainer.accumulate_grad_batches

    Dataset = datasets[hparams['dataset']['name']]
    batch_size = hparams['dataset']['batch_size'] // num_devices
    num_workers = os.cpu_count()

    dataset = {
        'train': Dataset(args.datadir, train=True,
                         transform=get_trfms(hparams['transform']['train'])),
        'valid': Dataset(args.datadir, train=False,
                         transform=get_trfms(hparams['transform']['valid'])),
    }
    dataloader = {
        'train': torch.utils.data.DataLoader(
            dataset['train'], batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True),
        'valid': torch.utils.data.DataLoader(
            dataset['valid'], batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True),
    }

    steps_per_epoch = len(dataloader['train']) // (num_accum * num_devices)

    model = Module(steps_per_epoch=steps_per_epoch, **hparams)
    trainer.fit(model, dataloader['train'], dataloader['valid'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('datadir', type=str)
    parser.add_argument('--random_seed', type=int, default=0)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    with open(args.config) as file:
        hparams = json.load(file)

    run(hparams, args)
