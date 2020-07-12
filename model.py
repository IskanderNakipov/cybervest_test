import argparse

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from dataset import SequenceDataset

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


class Model(nn.Module):
    def __init__(self, num_layers=2, hidden_size=128):
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_size, 2)

    def forward(self, X):
        X = self.rnn(X)[0]
        return torch.sigmoid(self.classifier(X))


def precision(pred, true):
    if pred.argmax(dim=1).sum().item() == 0:
        if true.sum().item() == 0:
            return 1
        else:
            return 0
    res = (((pred > 0.5).float() == true) & (true == 1)).float().sum() / (pred > 0.5).float().sum()
    return res.item()


def recall(pred, true):
    if true.sum().item() == 0:
        if ((pred > 0.5).float()).sum().item() == 0:
            return 1
        else:
            return 0
    res = (((pred > 0.5).float() == true) & (true == 1)).float().sum() / true.sum()
    return res.item()


def f1(pred, true):
    prec = precision(pred, true)
    rec = recall(pred, true)
    if prec * rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)


def _train_step(model, opt, x):
    X, YMin, YMax = x
    Y = torch.cat([YMin.unsqueeze(-1), YMax.unsqueeze(-1)], dim=-1).float()
    pred = model(X.unsqueeze(-1).float())
    print(pred.mean().item())
    loss = nn.functional.binary_cross_entropy(pred, Y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    min_f1 = f1(pred[:, :, 0], YMin)
    max_f1 = f1(pred[:, :, 1], YMax)
    return loss, min_f1, max_f1


def _eval(model, X_val, YMin_val, YMax_val, device):
    X_val, YMin_val, YMax_val = [torch.from_numpy(x).to(device) for x in [X_val, YMin_val, YMax_val]]
    Y = torch.cat([YMin_val.unsqueeze(-1), YMax_val.unsqueeze(-1)], dim=-1).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(X_val.unsqueeze(-1).unsqueeze(0).float())
        loss = nn.functional.binary_cross_entropy(pred, Y)
        min_f1 = f1(pred[:, :, 0], YMax_val)
        max_f1 = f1(pred[:, :, 1], YMin_val)
    return loss, min_f1, max_f1


def train(model: Model, X, X_val, YMin, YMin_val, YMax, YMax_val,
          N=1024, M=1000, num_epoch=10,
          epoch_size=1000, batch_size=32, device='cuda:0'):
    assert X.shape[0] == N*M
    train_dataset = SequenceDataset(X, YMin, YMax)
    model.to(device)
    opt = optim.Adam(model.parameters())  #TODO: add lr
    for epoch in range(num_epoch):
        with tqdm(total=epoch_size, desc=f'epoch {epoch}') as tq:
            model.train()
            indices = np.arange(0, N * (M - 1))
            indices = np.random.choice(indices, size=batch_size * epoch_size, replace=False)
            base_sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(train_dataset, sampler=base_sampler, batch_size=batch_size)
            for x in tqdm(train_loader):
                loss, min_f1, max_f1 = _train_step(model, opt, [x_.to(device) for x_ in x])
                tq.set_postfix(loss=loss.item())
                tq.update()
            model.eval()
        _eval(model, X_val, YMin_val, YMax_val, device)


def main():
    model = Model()
    args = parse_args()
    X = generate_x(args.M, args.N)
    YMin, YMax = find_min_max(X, args.T, args.k)

    X_val = generate_x(args.M, args.N)
    YMin_val, YMax_val = find_min_max(X_val, args.T, args.k)[:args.N]
    train(model, X, X_val, YMin, YMin_val, YMax, YMax_val, args.N, args.M)


main()