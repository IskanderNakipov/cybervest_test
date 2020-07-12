from helpers import parse_args, timer

from generate_data import generate_x, find_min_max


if __name__ == '__main__':
    filename = 'task_1.log'
    args = parse_args()
    x_msg = f"X generation with N = {args.N} and M = {args.M}"
    X = timer(generate_x, filename, x_msg)(args.M, args.N)
    y_msg = "Finding optimums for X"
    YMin, YMax = timer(find_min_max, filename, y_msg)(X, args.T, args.k)
