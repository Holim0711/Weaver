import os
from glob import glob
import imghdr
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


class PandasImageDataset(MyDataset):

    def __init__(self, df, img_trfm=None, trg_trfm=None):
        self.df = df
        self.img_trfm = img_trfm
        self.trg_trfm = trg_trfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, trg = self.df.iloc[idx]
        img = Image.open(path)
        if self.img_trfm:
            img = self.img_trfm(img)
        if self.trg_trfm:
            trg = self.trg_trfm(trg)
        return img, trg


class SimpleImageDataset(MyDataset):

    def __init__(self, root, trfm):
        filenames = glob(os.path.join(root, '**'), recursive=True)
        filenames = [x for x in filenames if os.path.isfile(x)]
        filenames = [x for x in filenames if imghdr.what(x)]
        self.filenames = sorted(filenames)
        self.trfm = trfm

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.trfm(Image.open(self.filenames[idx]))
