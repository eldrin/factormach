import numpy as np


def ndcg(actual, predicted, k=10):
    """ for binary relavance """
    if len(predicted) > k:
        predicted = predicted[:k]
    actual = set(actual)

    dcg = 0.
    idcg = 0.
    for i, p in enumerate(predicted):
        if p in actual:
            dcg += 1. / np.log2(i + 2.)
        if i < len(actual):
            idcg += 1. / np.log2(i + 2.)

    if len(actual) == 0:
        return None

    return dcg / idcg


def compute_ndcg(s, y_ts, y_tr, topk=100):
    """"""
    tr = y_tr.indices
    s[tr] = -float("inf")
    pred = s.argsort(descending=True)[:topk].cpu().data.numpy()
    true = y_ts.indices
    return ndcg(true, pred, topk)
