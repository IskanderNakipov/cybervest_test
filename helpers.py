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
    parser.add_argument('--T', type=float, help="Parameter T from task 1", default=2.89)
    parser.add_argument('--k', type=int, help="Parameter k from task 1", default=10)
    parser.add_argument('--amount_graphs', '-a', help='Amount of graphs to draw', default=10, type=int)
    parser.add_argument('--num_batches', '-nb', help='Amount of batches to sample every epoch', default=250, type=int)
    parser.add_argument('--batch_size', '-B', help='Batch size', default=64, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('--epochs', help='Number of epochs', default=10, type=int)
    parser.add_argument('--val_size', help="Size of validation data", default=1024*100, type=int)
    parser.add_argument('--device', help='Device to train on', default="cuda:0", type=str)
    parser.add_argument('--hidden_size', help='Size of hidden RNN layers', default=64, type=int)
    parser.add_argument('--num_layers', help='Size of hidden RNN layers', default=4, type=int)
    parser.add_argument('--model_path', help='Path to save or load model', default='./', type=str)
    return parser.parse_args()


def flatten(tensor):
    batch_size, length = tensor.shape[:2]
    return tensor.view(batch_size * length, -1)
