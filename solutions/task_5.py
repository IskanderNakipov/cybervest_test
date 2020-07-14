import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from generate_data import generate_x, find_min_max
from helpers import parse_args, timer
from model import Model
from visualisation import visualise_probas, plot_confusion_matrix

if __name__ == '__main__':
    filename = 'task_5.log'
    args = parse_args()
    x_msg = f"X generation with N = {args.N} and M = {args.M}"
    X = timer(generate_x, filename, x_msg)(args.M, args.N)
    y_msg = "Finding optimums for X"
    YMin, YMax = timer(find_min_max, filename, y_msg)(X, args.T, args.k)
    model = Model.load(args.model_path)
    probas = model.predict_proba(torch.from_numpy(X).unsqueeze(0).unsqueeze(-1).float()).numpy()
    for _ in range(args.amount_graphs):
        start = np.random.randint(0, args.N * (args.M - 1))
        visualise_probas(X, probas, start, args.N, YMin, YMax)
        plt.legend()
        plt.show()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).unsqueeze(0).unsqueeze(-1).float()).numpy()
        pred = pred.squeeze(0).argmax(axis=-1)
        Y = np.zeros_like(X)
        Y[YMin == 1] += 1
        Y[YMax == 1] += 2
        cm = confusion_matrix(Y, pred)
        print(classification_report(Y, pred))
    plot_confusion_matrix(cm, target_names=['None', 'Min', 'Max'])
