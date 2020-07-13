import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def visualise_x(X, start, N=1024, YMin=None, YMax=None):
    """
    Draws plot for array X[start:start + N]. If YMin and YMax are given,
    also puts dots in minimums and maximums.
    :param X: Array to visualise
    :param start: starting point
    :param N: length of visualisation
    :param YMin: Array with ones on minimums and zeros in other points
    :param YMax: Array with ones on maximums and zeros in other points
    :return: Axes
    """
    x = np.arange(start, start + N)
    ax = sns.lineplot(x, X[start: start + N])
    if YMin is not None:
        x = np.argwhere(YMin[start: start + N])[:, 0] + start
        plt.scatter(x, X[x], s=100, label='Min', c='pink')
    if YMax is not None:
        x = np.argwhere(YMax[start: start + N])[:, 0] + start
        plt.scatter(x, X[x], s=100, label='Max', c='lightgreen')
    plt.grid(True)
    plt.xlabel('Position in X')
    plt.ylabel("Value in X")
    return ax


def visualise_probas(X, probas, start, N=1024, YMin=None, YMax=None):
    """
    Draws plot for array X[start:start + N]. If YMin and YMax are given,
    also puts dots in minimums and maximums.
    Moreover, plots probabilities of minimum and maximum
    :param X: Array to visualise
    :param probas: array of point type probability
    :param start: starting point
    :param N: length of visualisation
    :param YMin: Array with ones on minimums and zeros in other points
    :param YMax: Array with ones on maximums and zeros in other points
    :return: Axes
    """
    ax = visualise_x(X, start, N, YMin, YMax)
    ax.twinx()
    x = np.arange(start, start + N)
    probas_ = probas[start: start + N]
    plt.stackplot(x, probas_[:, 0], alpha=0.25, color='red', labels=["Proba of minimum"])
    plt.stackplot(x, probas_[:, 1], alpha=0.25, color='green', labels=["Proba of maximum"])
    return ax


