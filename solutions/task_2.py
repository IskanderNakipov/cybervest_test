import numpy as np
from matplotlib import pyplot as plt

from helpers import parse_args, timer

from generate_data import generate_x, find_min_max
from visualisation import visualise_x


if __name__ == '__main__':
    filename = 'task_2.log'
    args = parse_args()
    x_msg = f"X generation with N = {args.N} and M = {args.M}"
    X = timer(generate_x, filename, x_msg)(args.M, args.N)
    y_msg = "Finding optimums for X"
    YMin, YMax = timer(find_min_max, filename, y_msg)(X, args.T, args.k)

    for _ in range(args.amount_graphs):
        start = np.random.randint(0, args.N * (args.M - 1))
        visualise_x(X, start, args.N, YMin, YMax)
        plt.legend()
        plt.show()

