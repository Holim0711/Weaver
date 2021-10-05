import os
from collections.abc import Iterable
from typing import Union, Callable
from torch.utils.data import Dataset, DataLoader
import torch

__all__ = [
    'SimpleDataset',
    'get_dataloader',
]


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = os.cpu_count(),
    pin_memory: bool = torch.cuda.is_available(),
    **kwargs
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )


class SimpleDataset(Dataset):
    def __init__(
        self,
        data: Union[Iterable, tuple[Iterable, Iterable]],
        transform_x: Callable = None,
        transform_y: Callable = None
    ):
        """ Simple Dataset

        Args
            - data: `x, y = data[i]` or `x, y = data[0][idx], data[1][idx]`
            - transform_x: `x = transform_x(x)`
            - transform_y: `y = transform_y(y)`
        """
        self.data = data
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data, tuple):
            x, y = self.data[0][idx], self.data[1][idx]
        else:
            x, y = self.data[idx]

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return x, y

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        **kwargs
    ):
        return get_dataloader(self, batch_size, shuffle, **kwargs)
