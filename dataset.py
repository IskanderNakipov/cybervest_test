import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class SequenceDataset(Dataset):
    """
    Dataset, which samples sequences with length N from every tensor from tensors
    :param tensors: list of tensor with same length
    :param N: length of sampled tensors
    """
    def __init__(self, *tensors: t.List[torch.Tensor], N: int = 1024):
        super().__init__()
        assert all([tensor.shape[0] == tensors[0].shape[0] for tensor in tensors])
        self.tensors = tensors
        self.N = N

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, i: int):
        tensors = [tensor[i:i + self.N] for tensor in self.tensors]
        return tensors


def make_data_loader(*tensors, N, batch_size, num_batches) -> torch.utils.data.DataLoader:
    """
    Creates DataLoader, which samples num_batches batches of sequences with shape (batch_size, N) from tensors
    :param tensors: tensors to sample batches from
    :param N: length of sampled subsequences
    :param batch_size: batch size
    :param num_batches: amount of batches to sample
    :return: DataLoader
    """
    assert tensors[0].shape[0] % N == 0, "length of tensors has to be dividable by N!"
    M = tensors[0].shape[0] // N
    dataset = SequenceDataset(*tensors, N=N)
    indices = np.arange(0, N * (M - 1))
    indices = np.random.choice(indices, size=batch_size * num_batches, replace=False)
    base_sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=base_sampler, batch_size=batch_size)
    return loader
