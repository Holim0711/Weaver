import os
import random
from PIL import Image
import torch

__all__ = [
    'Dataset',
    'RandomSubset',
    'SimpleDataset',
]


class Dataset(torch.utils.data.Dataset):
    def get_dataloader(
        self,
        batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        **kwargs
    ):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs)


class RandomSubset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.indices = random.sample(range(len(self.dataset)), n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class SimpleDataset(Dataset):
    def __init__(self, data, reader=None, transform=None, target_transform=None):
        """ Simple Image Dataset

        Args
            - data (list): `x, y = data[i]`
            - reader (func): `x = reader(root, x)`
            - transform (func): `x = transform(x)`
            - target_transform (func): `y = target_transform(y)`
        """
        self.data = data
        self.reader = reader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        x = x if self.reader is None else self.reader(x)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
