def precision(pred, true):
    if pred.argmax(dim=1).sum().item() == 0:
        if true.sum().item() == 0:
            return 1
        else:
            return 0
    res = (((pred > 0.5).float() == true) & (true == 1)).float().sum() / (pred > 0.5).float().sum()
    return res.item()


def recall(pred, true):
    if true.sum().item() == 0:
        if ((pred > 0.5).float()).sum().item() == 0:
            return 1
        else:
            return 0
    res = (((pred > 0.5).float() == true) & (true == 1)).float().sum() / true.sum()
    return res.item()


def f1(pred, true):
    prec = precision(pred, true)
    rec = recall(pred, true)
    if prec * rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)