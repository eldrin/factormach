import numpy as np
from scipy import sparse as sp
import torch


def scisp2tchsp(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def split_recsys_data(X, train_ratio=0.8, valid_ratio=0.1):
    """ Split given user-item matrix into train/test.

    This split is to check typical internal ranking accracy.
    (not checking cold-start problem)

    Inputs:
        X (scipy.sparse.csr_matrix): user-item matrix
        train_ratio (float): ratio of training records per user
        test_ratio (float): ratio of validation records per user

    Returns:
        scipy.sparse.csr_matrix: training matrix
        scipy.sparse.csr_matrix: validation matrix
        scipy.sparse.csr_matrix: testing matrix
    """
    def _store_data(cur_i, container, indices, data, rnd_idx, start, end):
        n_records = end - start
        if n_records == 0:
            return
        container['I'].extend(np.full((end - start,), cur_i).tolist())
        container['J'].extend(indices[rnd_idx[start:end]].tolist())
        container['V'].extend(data[rnd_idx[start:end]].tolist())

    def _build_mat(container, shape):
        return sp.coo_matrix(
            (container['V'], (container['I'], container['J'])),
            shape=shape
        ).tocsr()

    # prepare empty containers
    train = {'V': [], 'I': [], 'J': []}
    valid = {'V': [], 'I': [], 'J': []}
    test = {'V': [], 'I': [], 'J': []}
    for i in range(X.shape[0]):
        idx, dat = slice_row_sparse(X, i)
        rnd_idx = np.random.permutation(len(idx))
        n = len(idx)
        train_bound = int(train_ratio * n)
        if np.random.rand() > 0.5:
            valid_bound = int(valid_ratio * n) + train_bound
        else:
            valid_bound = int(valid_ratio * n) + train_bound + 1

        _store_data(i, train, idx, dat, rnd_idx, 0, train_bound)
        _store_data(i, valid, idx, dat, rnd_idx, train_bound, valid_bound)
        _store_data(i, test, idx, dat, rnd_idx, valid_bound, n)

    return tuple(
        _build_mat(container, X.shape)
        for container in [train, valid, test]
    )


def load_triplet_csv(fn):
    """
    """
    pass
