import torch

from model import Model, train
from generate_data import generate_x, find_min_max
from helpers import parse_args, timer


if __name__ == '__main__':
    model = Model()
    args = parse_args()
    filename = "task_4.log"
    x_msg = f"X generation with N = {args.N} and M = {args.M}"
    X = timer(generate_x, filename, x_msg)(args.M, args.N)
    y_msg = "Finding optimums for X"
    YMin, YMax = timer(find_min_max, filename, y_msg)(X, args.T, args.k)

    X_val = generate_x(args.M, args.N)
    YMin_val, YMax_val = find_min_max(X_val, args.T, args.k)[:args.N]
    X, X_val, YMin, YMin_val, YMax, YMax_val = [torch.from_numpy(x) for x in [X, X_val, YMin, YMin_val, YMax, YMax_val]]
    train(model, X, X_val, YMin, YMin_val, YMax, YMax_val, args.N, args.M, args.epochs, args.lr,
          args.num_batches, args.batch_size, args.device)

