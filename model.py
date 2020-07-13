import argparse
import typing as t

import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import make_data_loader
from generate_data import generate_x, find_min_max
from metrics import f1
from helpers import flatten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help="Parameter N from task 1", default=1024)
    parser.add_argument('--M', type=int, help="Parameter M from task 1", required=True)
    parser.add_argument('--T', type=int, help="Parameter T from task 1", default=3)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    parser.add_argument('--num_batches', '-nb', help='Amount of batches to sample every epoch', default=100)
    parser.add_argument('--batch_size', '-B', help='Batch size', default=16)
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, num_layers=2, hidden_size=128):
        """
        BiGRU neural model, which finds minimums and maximums of time series with
        :param num_layers: number of GRU layers
        :param hidden_size: size of hidden GRU layers
        """
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_size, 3)

    def forward(self, X):
        X = self.rnn(X)[0]
        return self.classifier(X)


def _unpack_data(x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpacks data to suitable for neural network form
    :param x: tuple [input, YMin, YMax]
    :return: training data and labels
    """
    X, YMin, YMax = x
    placeholder = torch.ones_like(YMin) * 0.5
    Y = torch.cat([placeholder.unsqueeze(-1),
                   YMin.unsqueeze(-1),
                   YMax.unsqueeze(-1)], dim=-1)
    Y = Y.argmax(dim=-1)
    return X, Y


def _train_step(model: Model, opt: optim.Optimizer,
                x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, float, float]:
    """
    Performs one step of training
    :param model: model to train
    :param opt: optimizer
    :param x: tuple [input, YMin, YMax]
    :return: loss, f1-score for minimums anf f1-score for maximums
    """
    X, Y = _unpack_data(x)
    YMin, YMax = x[1:]
    pred = model(X.unsqueeze(-1).float())
    loss = nn.functional.cross_entropy(flatten(pred), Y.flatten())
    loss.backward()
    opt.step()
    opt.zero_grad()
    min_f1 = f1(flatten(pred), YMin.flatten())  # TODO: fix pred indexing
    max_f1 = f1(flatten(pred), YMax.flatten())
    return loss, min_f1, max_f1


def _eval(model: Model, x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, float, float]:
    """
    Evaluates model
    :param model: model to eval
    :param x: tuple [input, YMin, YMax]
    :return: loss, f1-score for minimums anf f1-score for maximums
    """
    X, Y = _unpack_data(x)
    YMin, YMax = x[1:]
    with torch.no_grad():
        pred = model(X.unsqueeze(-1).unsqueeze(0).float())
        loss = nn.functional.cross_entropy(flatten(pred), Y.flatten())
        min_f1 = f1(flatten(pred), YMin.flatten())
        max_f1 = f1(flatten(pred), YMax.flatten())
    return loss, min_f1, max_f1


def train(model: Model, X: torch.Tensor, X_val: torch.Tensor,
          YMin: torch.Tensor, YMin_val: torch.Tensor,
          YMax: torch.Tensor, YMax_val: torch.Tensor,
          N: int = 1024, M: int = 1000, num_epoch: int = 10, lr: float = 1e-3,
          epoch_size: int = 1000, batch_size: int = 32, device: str = 'cuda:0') -> Model:
    """
    Trains model
    :param model: model to train
    :param X: Training data
    :param X_val: validation data
    :param YMin: Training labels for minimums
    :param YMin_val: validation labels for minimums
    :param YMax: Training labels for maximums
    :param YMax_val: validation labels for maximums
    :param N: length of subsequences in data
    :param M: amount of subsequences in data
    :param num_epoch: amount of epochs to train model
    :param lr: learning rate
    :param epoch_size: amount of batches to feed every epoch
    :param batch_size: batch size
    :param device: device
    :return: trained model
    """
    assert X.shape[0] == N*M
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epoch):
        with tqdm(total=epoch_size, desc=f'epoch {epoch}') as tq:
            model.train()
            train_loader = make_data_loader(X, YMin, YMax, N=N, batch_size=batch_size, num_batches=epoch_size)
            for x in tqdm(train_loader):
                loss, min_f1, max_f1 = _train_step(model, opt, [x_.to(device) for x_ in x])
                tq.set_postfix(loss=loss.item())
                tq.update()
            model.eval()
        _eval(model, [x_.to(device) for x_ in [X_val, YMin_val, YMax_val]])
    return model


def main():
    model = Model()
    args = parse_args()
    X = generate_x(args.M, args.N)
    YMin, YMax = find_min_max(X, args.T, args.k)

    X_val = generate_x(args.M, args.N)
    YMin_val, YMax_val = find_min_max(X_val, args.T, args.k)[:args.N]
    train(model, X, X_val, YMin, YMin_val, YMax, YMax_val, args.N, args.M)


main()