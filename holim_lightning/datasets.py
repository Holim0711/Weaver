import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def get_dataloader(self, batch_size, num_workers=None, pin_memory=True, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers if num_workers else os.cpu_count(),
            pin_memory=pin_memory,
            **kwargs)


class SimpleImageDataset(MyDataset):

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


class PandasImageDataset(MyDataset):

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
