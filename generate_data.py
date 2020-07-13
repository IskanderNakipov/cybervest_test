import numpy as np


def generate_x(M: int, N: int = 1024):
    """Generates array X with shape (M*N, )"""
    X = 2 * (np.random.rand(M * N) - 0.5)
    X[0] = 1
    X = np.cumsum(X)
    return X


def find_min_max(X, T=3, k=10):
    """Finds local optimums for X"""
    #TODO: replace placeholder with real function
    min_dist = T*np.std(X[:-k] - X[k:])
    YMin = np.zeros_like(X)
    YMax = np.zeros_like(X)
    YMin[list(range(0, X.shape[0], 200))] += 1
    YMax[list(range(2, X.shape[0], 200))] += 1
    return YMin, YMax


