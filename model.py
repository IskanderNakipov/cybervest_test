import logging
import typing as t

import torch
from sru import SRU
from sklearn.metrics import classification_report
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from dataset import make_data_loader
from helpers import flatten


class Model(nn.Module):
    def __init__(self, num_layers=4, hidden_size=512):
        """
        BiGRU neural model, which finds minimums and maximums of time series with
        :param num_layers: number of GRU layers
        :param hidden_size: size of hidden GRU layers
        """
        super().__init__()
        self.rnn = SRU(1, hidden_size, num_layers, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden_size, 3)

    def forward(self, X):
        X -= X.roll(1, 1)
        X[:, 0] *= 0
        X = self.rnn(X.transpose(1, 0))[0].transpose(1, 0)
        return self.classifier(X)

    def predict_proba(self, X):
        with torch.no_grad():
            return self(X).squeeze(0)[:, 1:]


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
                x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs one step of training
    :param model: model to train
    :param opt: optimizer
    :param x: tuple [input, YMin, YMax]
    :return: loss, predicts and true labels
    """
    X, Y = _unpack_data(x)
    pred = model(X.unsqueeze(-1).float())
    loss = nn.functional.cross_entropy(flatten(pred), Y.flatten(),
                                       weight=torch.Tensor([2e-2, 1, 1]).to(pred.device))
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss, flatten(pred).argmax(dim=-1), Y.flatten()


def _eval(model: Model, x: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates model
    :param model: model to eval
    :param x: tuple [input, YMin, YMax]
    :return: loss, predicts and true labels
    """
    X, Y = _unpack_data(x)
    with torch.no_grad():
        pred = model(X.unsqueeze(-1).unsqueeze(0).float())
        loss = nn.functional.cross_entropy(flatten(pred), Y.flatten(),
                                           weight=torch.Tensor([2e-2, 1, 1]).to(pred.device))
    return loss, flatten(pred).argmax(dim=-1), Y.flatten()


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
    sh = optim.lr_scheduler.StepLR(opt, 1, 0.9)
    for epoch in range(num_epoch):
        with tqdm(total=epoch_size, desc=f'epoch {epoch} of {num_epoch}') as tq:
            model.train()
            train_loader = make_data_loader(X, YMin, YMax, N=N, batch_size=batch_size, num_batches=epoch_size)
            pred, true = [], []
            for x in tqdm(train_loader):
                loss, pred_, true_ = _train_step(model, opt, [x_.to(device) for x_ in x])
                pred.append(pred_.detach().cpu().numpy())
                true.append(true_.cpu().numpy())
                tq.set_postfix(loss=loss.item(), lr=sh.get_last_lr())
                tq.update()
            sh.step()
            true = np.concatenate(true)
            pred = np.concatenate(pred)
            print(classification_report(true, pred, labels=[1, 2], target_names=['Min', 'Max']))
            scores = classification_report(true, pred, labels=[1, 2], target_names=['Min', 'Max'], output_dict=True)['micro avg']
            del scores['support']
            logging.info(f"Training for epoch {epoch} ended, scores are {scores}")

        model.eval()
        loss, pred, true = _eval(model, [x_.to(device) for x_ in [X_val, YMin_val, YMax_val]])
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        print(classification_report(true, pred, labels=[1, 2], target_names=['Min', 'Max']))
        scores = classification_report(true, pred, labels=[1, 2], target_names=['Min', 'Max'], output_dict=True)['micro avg']
        del scores['support']
        scores['loss'] = loss.item()
        logging.info(f"Validation for epoch {epoch} ended, scores are {scores}")
    return model
