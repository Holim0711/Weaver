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


class RandomSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        length: Union[int, float],
        random_seed: int = 0,
    ):
        if length < 1.0:
            length = round(len(dataset) * length)
        self.length = length
        self.random_state = np.random.RandomState(random_seed)
        indices = self.random_state.choice(len(dataset), length, replace=False)
        super().__init__(dataset, indices)


class FewShotSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        k_shots: int,
        random_seed: int = 0,
    ):
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        else:
            targets = np.array([y for x, y in dataset])
        self.n_ways = targets.max() + 1
        self.k_shots = k_shots
        self.random_state = np.random.RandomState(random_seed)
        indices = []
        for c in range(self.n_ways):
            c_idxs = np.where(targets == c)[0]
            c_idxs = self.random_state.choice(c_idxs, k_shots, replace=False)
            indices.extend(c_idxs)
        super().__init__(dataset, indices)
