import argparse

from time import time
import logging

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help="Parameter N from task 1", default=1024)
    parser.add_argument('--M', type=int, help="Parameter M from task 1", required=True)
    parser.add_argument('--T', type=int, help="Parameter T from task 1", default=3)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    return parser.parse_args()


def generate_x(M: int, N: int = 1024):
    """Generates array X with shape (M*N, )"""
    start = time()
    logging.info(f"X generation with N = {N} and M = {M}")
    X = 2 * (np.random.rand(M * N) - 0.5)
    X[0] = 1
    X = np.cumsum(X)
    end = time()
    logging.info(f"X generation ended and took {end - start} seconds")
    return X


def find_min_max(X, T=3, k=10):
    """Finds local optimums for X"""
    #TODO: replace placeholder with real function
    start = time()
    logging.info(f"Finding optimums for X started")
    min_dist = T*np.std(X[:-k] - X[k:])
    YMin = np.zeros_like(X)
    YMax = np.zeros_like(X)
    YMin[list(range(0, X.shape[0], 5 * X.shape[0] // 1024))] += 1
    YMax[list(range(2, X.shape[0], 5 * X.shape[0] // 1024))] += 1
    end = time()
    logging.info(f"Finding optimums ended and took {end - start} seconds")
    return YMin, YMax


if __name__ == '__main__':
    logging.basicConfig(filename='task_1.log', filemode='w',
                        format="%(asctime)s--%(levelname)s--%(message)s", level=logging.INFO)
    args = parse_args()
    X = generate_x(args.M, args.N)
    YMin, YMax = find_min_max(X, args.T, args.k)
