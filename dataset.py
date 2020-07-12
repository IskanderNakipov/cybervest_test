from time import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import argparse
import logging

from generate_data import generate_x, find_min_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help="Parameter N from task 1", default=1024)
    parser.add_argument('--M', type=int, help="Parameter M from task 1", required=True)
    parser.add_argument('--T', type=int, help="Parameter T from task 1", default=3)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    parser.add_argument('--num_batches', '-nb', help='Amount of batches to sample', default=100)
    parser.add_argument('--batch_size', '-B', help='Batch size', default=16)
    return parser.parse_args()


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


if __name__ == '__main__':
    logging.basicConfig(filename='task_3.log', filemode='w',
                        format="%(asctime)s--%(levelname)s--%(message)s", level=logging.INFO)
    args = parse_args()
    X = generate_x(args.M, args.N)
    YMin, YMax = find_min_max(X, args.T, args.k)

    dataset = SequenceDataset(X, YMin, YMax, args.N)
    indices = np.arange(0, args.N * (args.M - 1))
    indices = np.random.choice(indices, size=args.batch_size * args.num_batches, replace=False)
    base_sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=base_sampler, batch_size=16)
    start = time()
    logging.info(f"{args.num_batches} batches sampling with batch size = {args.batch_size} started")
    for i, x in enumerate(loader):
        pass
    logging.info(f"Batch sampling ended and took {time() - start} seconds")
