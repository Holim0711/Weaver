import os
import imghdr
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import pandas as pd
from .datasets import PandasImageDataset


def split_foreach(root, test_size):
    train_split, valid_split = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        filenames = [os.path.join(dirpath, x) for x in filenames]
        filenames = [x for x in filenames if imghdr.what(x)]
        if filenames:
            if len(filenames) > 1:
                train, valid = train_test_split(filenames, test_size=test_size)
                train_split += train
                valid_split += valid
            else:
                valid_split += filenames
    return train_split, valid_split


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, prep_dir, img_dir_list, test_size,
                 train_trfm, valid_trfm, target_trfm=None):
        super().__init__()

        if all(isinstance(x, str) for x in img_dir_list):
            img_dir_list = [[x] for x in img_dir_list]

        assert all(isinstance(y, str) for x in img_dir_list for y in x)

        if not os.path.isdir(prep_dir):
            os.makedirs(prep_dir)

        self.batch_size = batch_size
        self.prep_dir = prep_dir
        self.img_dir_list = img_dir_list
        self.test_size = test_size
        self.train_trfm = train_trfm
        self.valid_trfm = valid_trfm
        self.target_trfm = target_trfm
        self.num_workers = os.cpu_count()
        self.pin_memory = True

    def prepare_data(self):
        if os.path.isfile(os.path.join(self.prep_dir, 'meta.txt')):
            return

        train_split, valid_split = [], []

        for cls, img_dirs in enumerate(self.img_dir_list):
            for img_dir in img_dirs:
                train, valid = split_foreach(img_dir, self.test_size)
                train_split += [(x, cls) for x in train]
                valid_split += [(x, cls) for x in valid]

        with open(os.path.join(self.prep_dir, 'train_split.tsv'), 'w') as file:
            for path, cls in train_split:
                print(path, cls, sep='\t', file=file)

        with open(os.path.join(self.prep_dir, 'valid_split.tsv'), 'w') as file:
            for path, cls in valid_split:
                print(path, cls, sep='\t', file=file)

        with open(os.path.join(self.prep_dir, 'meta.txt'), 'w') as file:
            print(f"test_size: {self.test_size}", file=file)
            print("tsv_header: path, class", file=file)
            print("--- source directories ---", file=file)
            for cls, img_dirs in enumerate(self.img_dir_list):
                for x in img_dirs:
                    print(f"{cls}\t{x}", file=file)

    def setup(self, stage=None):
        train = pd.read_csv(os.path.join(self.prep_dir, 'train_split.tsv'), sep='\t', header=None)
        valid = pd.read_csv(os.path.join(self.prep_dir, 'valid_split.tsv'), sep='\t', header=None)
        self.train_data = PandasImageDataset('.', train, self.train_trfm, self.target_trfm)
        self.valid_data = PandasImageDataset('.', valid, self.valid_trfm, self.target_trfm)

    def train_dataloader(self):
        return self.train_data.get_dataloader(
            self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return self.valid_data.get_dataloader(
            self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
