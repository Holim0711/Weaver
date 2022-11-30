from typing import Union
from torch.utils.data import Dataset, Subset
import numpy as np


__all__ = ['IndexedDataset', 'RandomSubset']


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def balanced_random_select(targets, length, random_state):
    targets = np.array(targets)
    classes = np.unique(targets)
    n = length // len(classes)

    results = []
    for c in classes:
        indices = np.where(targets == c)[0]
        indices = random_state.choice(indices, n, replace=False)
        results.extend(indices)
    return results


class RandomSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        length: Union[int, float],
        class_balanced: bool = False,
        random_seed: int = 0,
    ):
        if length < 1.0:
            length = round(len(dataset) * length)

        random_state = np.random.RandomState(random_seed)
        if not class_balanced:
            indices = random_state.choice(len(dataset), length, replace=False)
        else:
            targets = [int(y) for _, y in dataset]
            indices = balanced_random_select(targets, length, random_state)
        super().__init__(dataset, indices)
