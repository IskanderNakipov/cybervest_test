import argparse
import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from generate_data import generate_x, find_min_max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help="Parameter N from task 1", default=1024)
    parser.add_argument('--M', type=int, help="Parameter M from task 1", required=True)
    parser.add_argument('--T', type=int, help="Parameter T from task 1", default=3)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    return parser.parse_args()


def visualise_x(X, start, N=1024, YMin=None, YMax=None):
    """
    Draws plot for array X[start:start + N]. If YMin and YMax are given,
    also puts dots in minimums and maximums.
    :param X: Array to visualise
    :param start: starting point
    :param N: length of visualisation
    :param YMin: Array with ones on minimums and zeros in other points
    :param YMax: Array with ones on maximums and zeros in other points
    :return: None
    """
    x = np.arange(start, start + N)
    sns.lineplot(x, X[start: start + N])
    if YMin is not None:
        x = np.argwhere(YMin[start: start + N])[:, 0] + start
        sns.scatterplot(x, X[x])
    if YMax is not None:
        x = np.argwhere(YMax[start: start + N])[:, 0] + start
        sns.scatterplot(x, X[x])
    plt.grid(True)
    plt.xlabel('Position in X')
    plt.ylabel("Value in X")


if __name__ == '__main__':
    logging.basicConfig(filename='task_2.log', filemode='w',
                        format="%(asctime)s--%(levelname)s--%(message)s", level=logging.INFO)
    args = parse_args()
    X = generate_x(args.M, args.N)
    YMin, YMax = find_min_max(X, args.T, args.k)

    for _ in range(args.amount):
        start = np.random.randint(0, args.N * (args.M - 1))
        visualise_x(X, start, args.N, YMin, YMax)
        plt.show()
