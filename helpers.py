import argparse

import logging
from time import time


def timer(func, filename, msg):
    logging.basicConfig(filename=filename, filemode='w',
                        format="%(asctime)s--%(levelname)s--%(message)s", level=logging.INFO)
    def wrapper(*args, **kwargs):
        start = time()
        logging.info(f"{msg} started")
        res = func(*args, **kwargs)
        logging.info(f"{msg} ended and took {time() - start} seconds")
        return res
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, help="Parameter N from task 1", default=1024)
    parser.add_argument('--M', type=int, help="Parameter M from task 1", required=True)
    parser.add_argument('--T', type=int, help="Parameter T from task 1", default=3)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    parser.add_argument('--amount_graphs', '-a', help='Amount of graphs to draw', default=10)
    parser.add_argument('--num_batches', '-nb', help='Amount of batches to sample', default=100)
    parser.add_argument('--batch_size', '-B', help='Batch size', default=16)
    return parser.parse_args()


def flatten(tensor):
    batch_size, length = tensor.shape[:2]
    return tensor.view(batch_size * length, -1)
