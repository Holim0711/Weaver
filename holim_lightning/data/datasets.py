import os
import random
from PIL import Image
import torch


class Dataset(torch.utils.data.Dataset):
    def get_dataloader(self, batch_size, shuffle=False, num_workers=None, pin_memory=True, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers else os.cpu_count(),
            pin_memory=pin_memory,
            **kwargs)


class RandomSampleDataset(Dataset):

    def __init__(self, dataset, n):
        self.dataset = dataset
        self.indices = random.sample(range(len(self.dataset)), n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class SimpleImageDataset(Dataset):

    def __init__(self, root, data, transform=None, target_transform=None):
        self.root = root
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = Image.open(os.path.join(self.root, x))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


class PandasImageDataset(Dataset):

    def __init__(self, root, df, transform=None, target_transform=None):
        self.root = root
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x, y = self.df.iloc[idx]
        x = Image.open(os.path.join(self.root, x))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
