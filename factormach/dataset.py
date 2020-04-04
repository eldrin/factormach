import numpy as np
import numba as nb

import torch
from torch.utils.data import Dataset


class RecFeatDataset(Dataset):
    def __init__(self, user_item, user_feature=None, item_feature=None, n_negs=10):
        super().__init__()
        self.user_item = user_item
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.n_negs = n_negs

    def _draw_data(self, u, user_item):
        i0, i1 = user_item.indptr[u], user_item.indptr[u+1]
        pos = user_item.indices[i0:i1]
        if len(pos) == 0:
            return None, None
        else:
            j0 = np.random.choice(len(pos))
            negs = negsamp_vectorized_bsearch(
                pos, user_item.shape[1], n_samp=self.n_negs
            )
            return pos[j0][None], negs

    def _preproc(self, u, pos, negs):
        """"""
        u = torch.full((self.n_negs + 1,), u).long()
        i = torch.LongTensor(np.r_[pos, negs])
        y = torch.LongTensor([1] + [-1] * self.n_negs)
        return u[:, None], i[:, None], y

    def __len__(self):
        return self.user_item.shape[0]

    def __getitem__(self, u_idx):
        pos, negs = self._draw_data(u_idx, self.user_item)
        if pos is None:
            return None, None, None, None, None

        u, i, y = self._preproc(u_idx, pos, negs)
        xi = torch.FloatTensor([-1])
        xu = torch.FloatTensor([-1])
        if self.user_feature is not None:
            xu = torch.FloatTensor(self.user_feature[u[:, 0]])
        if self.item_feature is not None:
            xi = torch.FloatTensor(self.item_feature[i[:, 0]])
        return u, i, y, xu, xi


def collate_triplets_with_feature(samples):
    """"""
    return tuple(
        map(
            torch.cat,
            zip(*[s for s in samples if s[0] is not None])
        )
    )


@nb.njit("i8[:](i4[:], i8, i8)")
def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    ss = np.searchsorted(pos_inds_adj, raw_samp, side='right')
    neg_inds = raw_samp + ss
    return neg_inds
