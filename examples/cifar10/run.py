import os
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchmetrics import Accuracy
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite import seed_everything
from weaver import get_classifier, get_optimizer, get_scheduler, get_transforms


class Module(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_classifier(**self.hparams.model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def shared_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        acc = self.train_acc if self.training else self.val_acc
        acc.update(z.softmax(dim=-1), y)
        return {'loss': loss}

    def shared_epoch_end(self, outputs):
        phase = 'train' if self.training else 'val'
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'{phase}/loss', loss)
        acc = self.train_acc if self.training else self.val_acc
        self.log(f'{phase}/acc', acc.compute())
        acc.reset()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs)

    def configure_optimizers(self):
        optim = get_optimizer(self.parameters(), **self.hparams.optimizer)
        sched = get_scheduler(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}


def run(args):
    hparams = json.load(open(args.config))
    seed_everything(hparams.get('random_seed', 0))

    callbacks = [ModelCheckpoint(save_top_k=0), LearningRateMonitor()]
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    train_transform = Compose(get_transforms(hparams['transform']['train']))
    val_transform = Compose(get_transforms(hparams['transform']['val']))

    dataset_root = hparams['dataset']['root']
    train_dataset = CIFAR10(dataset_root, transform=train_transform)
    val_dataset = CIFAR10(dataset_root, train=False, transform=val_transform)

    batch_size = hparams['dataset']['batch_size']
    num_workers = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    model = Module(**hparams)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run(args)
