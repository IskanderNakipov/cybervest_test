import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class SequenceDataset(Dataset):
    def __init__(self, X, YMin, YMax, N=1024):
        super().__init__()
        self.X = torch.from_numpy(X)
        self.YMin = torch.from_numpy(YMin)
        self.YMax = torch.from_numpy(YMax)
        self.N = N

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        X = self.X[i: i + self.N]
        YMin = self.YMin[i: i + self.N]
        YMax = self.YMax[i: i + self.N]
        return X, YMin, YMax


def make_data_loader(X, YMin, YMax, N, M, batch_size, num_batches):
    dataset = SequenceDataset(X, YMin, YMax, N)
    indices = np.arange(0, N * (M - 1))
    indices = np.random.choice(indices, size=batch_size * num_batches, replace=False)
    base_sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=base_sampler, batch_size=batch_size)
    return loader
