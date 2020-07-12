import numpy as np
from dataset import make_data_loader
from helpers import parse_args, timer
from generate_data import generate_x, find_min_max


def sample(loader):
    for x in loader:
        pass


if __name__ == '__main__':
    args = parse_args()
    filename = 'task_3.log'
    x_msg = f"X generation with N = {args.N} and M = {args.M}"
    X = timer(generate_x, filename, x_msg)(args.M, args.N)
    y_msg = "Finding optimums for X"
    YMin, YMax = timer(find_min_max, filename, y_msg)(X, args.T, args.k)

    loader = make_data_loader(X, YMin, YMax, args.N, args.M, args.batch_size, args.num_batches)
    timer(sample, filename,
          f"{args.num_batches} batches sampling with batch size = {args.batch_size}")(loader)
