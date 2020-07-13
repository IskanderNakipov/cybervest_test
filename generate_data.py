import numpy as np
from numba import njit, jit


def generate_x(M: int, N: int = 1024):
    """Generates array X with shape (M*N, )"""
    X = 2 * (np.random.rand(M * N) - 0.5)
    X[0] = 1
    X = np.cumsum(X)
    return X


def find_min_max(X, T=3, k=10):
    """Finds local optimums for X"""
    min_dist = T*np.std(X[:-k] - X[k:])
    max_index = X.argmax()

    indexes = np.zeros_like(X)
    indexes[max_index] += 1
    _find_min_max(X, min_dist, max_index, indexes)
    _find_min_max(X, min_dist, max_index, indexes, step=-1)
    YMax = np.zeros_like(X)
    YMax[indexes == 1] += 1
    YMin = np.zeros_like(X)
    YMin[indexes == -1] += 1
    return YMin, YMax


@jit(nopython=True)
def _find_min_max(X, min_dist, start, indexes, step=1):
    index = start
    end = X.shape[0] if step > 0 else -1
    candidate = -X.shape[0]
    on_max = 1
    for i in range(start, end, step):
        if candidate == -X.shape[0] and (X[index] - X[i]) * on_max >= min_dist:
            candidate = i
            continue
        if candidate != -X.shape[0]:
            if X[i] * on_max <= X[candidate] * on_max:
                candidate = i
            elif (X[i] - X[candidate]) * on_max >= min_dist:
                indexes[candidate] -= on_max
                index = candidate
                candidate = i
                on_max *= -1
    if candidate is not None:
        indexes[candidate] -= on_max
    else:
        indexes[end] -= on_max
    return indexes
