import datetime
import numpy as np
from scipy import sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..dataset import RecFeatDataset, collate_triplets_with_feature
from ..optimizer import MultipleOptimizer
from ..utils import scisp2tchsp
from ..metric import compute_ndcg


# TODO: make decorator for checking the model is already fitted
class UserItemFM(nn.Module):
    def __init__(self, k, init=0.001, n_iters=10, learn_rate=0.001, l2=0.0001,
                 n_negs=10, use_gpu=False, loss='sgns', loss_agg="sum",
                 alpha=None, no_item_factor=False, warm_start=False):
        """"""
        super().__init__()
        self.k = k
        self.init = init
        self.learn_rate = learn_rate
        self.l2 = l2
        self.n_iters = n_iters
        self.n_negs = n_negs
        # TODO: generalization of the device selection (not only for cuda)
        self.target_device = 'cuda' if use_gpu else 'cpu'
        self.no_item_factor = no_item_factor
        self.warm_start = warm_start

        # setup the loss
        if loss not in {'sgns', 'bpr', 'bce', 'kl'}:
            raise ValueError(
                '[ERROR] Loss should be one of {"sgns", "bpr", "bce", "kl"}'
            )
        self.loss = loss  # {'sgns', 'bpr', 'bce', 'kl', 'mse'}
        self.full_batch = True if loss in {'bce', 'kl'} else False
        if loss == 'bpr' and n_negs > 1:
            print(
                '[Warning] BPR loss only samples 1 negative ' +
                'sample! fall back to n_negs=1...'
            )
            self.n_negs = 1

        self.loss_agg = loss_agg
        self.alpha = alpha  # for weighted loss

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_embeddings(self, user_item, user_feature=None, item_feature=None):
        """"""
        # initialize the factors
        self.is_features_ = (
            False
            if (user_feature is None) and (item_feature is None)
            else True
        )
        item_sparse = True if self.loss in {'sgns', 'bpr'} else False
        self.register_parameter('w0', nn.Parameter(torch.FloatTensor([0])))
        self.embeddings_ = nn.ModuleDict({
            'user': nn.Embedding(user_item.shape[0], self.k + 1, sparse=True),
            'item': nn.Embedding(user_item.shape[1],
                                 self.k + 1, sparse=item_sparse),
        })
        if user_feature is not None:
            self.embeddings_['feat_user'] = nn.Embedding(
                user_feature.shape[1], self.k + 1,
                sparse=sp.issparse(user_feature)
            )
        elif item_feature is not None:
            self.embeddings_['feat_item'] = nn.Embedding(
                item_feature.shape[1], self.k + 1,
                sparse=sp.issparse(item_feature)
            )

        for key, layer in self.embeddings_.items():
            if self.no_item_factor and key == 'item':
                layer.weight.data.zero_()
                layer.weight.requires_grad = False
            else:
                layer.weight.data[:,:-1].normal_(std=self.init)
                layer.weight.data[:,-1].zero_()

    def _init_optimizer(self):
        """
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        params = {'sparse':[], 'dense':[]}
        for lyr in self.embeddings_.children():
            if lyr.sparse:
                params['sparse'].append(lyr.weight)
            else:
                params['dense'].append(lyr.weight)

        # determine if sparse optimizer needed
        if len(params['sparse']) == 0:
            self.opt = optim.Adam(
                params['dense'], lr=self.learn_rate, weight_decay=self.l2
            )
        elif len(params['dense']) == 0:
            self.opt = optim.SparseAdam(params['sparse'], lr=self.learn_rate)
        else:
            # init multi-optimizers
            # register it to the instance
            self.opt = MultipleOptimizer(
                optim.SparseAdam(params['sparse'], lr=self.learn_rate),
                optim.Adam(params['dense'], 
                           lr=self.learn_rate,
                           weight_decay=self.l2)
            )

    def _retrieve_factors(self, u, i, feats=None):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        # retrieve embeddins
        user = self.embeddings_['user'](u)
        item = self.embeddings_['item'](i)
        # post-process (devide factor and weight)
        w = torch.cat([user[..., -1], item[..., -1]], dim=1)
        v = torch.cat([user[..., :-1], item[..., :-1]], dim=1)
        for entity, feat in feats.items():
            k = 'feat_{}'.format(entity)
            if k in self.embeddings_:
                if self.embeddings_[k].sparse:
                    feat = self.embeddings_[k](feats)
                else:
                    feat = (
                        self.embeddings_[k].weight[None] * feat[..., None]
                    )
                # post-process (devide factor and weight)
                w = torch.cat([w, feat[..., -1]], dim=1)
                v = torch.cat([v, feat[..., :-1]], dim=1)
        return w, v

    def _update_z(self, features):
        """
        this method pre-compute & cache item-tag factor for prediction
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        self.cpu()  # since below process eats up lots of memory

        # update z
        for entity in ['user', 'item']:
        # for entity, feat in features.items():
            v = self.embeddings_[entity].weight.detach()
            k = 'feat_{}'.format(entity)

            if entity in features:
            # if k in self.embeddings_:
                feat = features[entity].to('cpu')
                vf = self.embeddings_[k].weight.detach()
                zf = feat @ vf
                zf2 = feat**2 @ vf**2
                z = (v + zf)
                z2 = (v**2 + zf2)
            else:
                z, z2 = v, (v**2)

            # register
            self.register_buffer('z{}'.format(entity[0]), z)
            self.register_buffer('z{}2'.format(entity[0]), z2)
        self.to(self.target_device)

    def predict_user(self, users):
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        if np.isscalar(users):
            users = [users]

        if not self.training:
            s, _ = self.forward_batch(users, None)
            return s
        else:
            raise ValueError('You should change to the .eval() mode first!')

    def forward(self, u, i, feats):
        """
        u (torch.LongTensor): user ids
        i (torch.LongTensor): item ids
        feats (torch.FloatTensor): item feature tensor
        """
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        w, v = self._retrieve_factors(u, i, feats)
        w_ = w.sum(1)
        v_ = (v.sum(1)**2 - (v**2).sum(1)).sum(-1) * .5
        s = self.w0 + w_ + v_
        return s, [w, v]

    def forward_batch(self, u, X=None):
        """"""
        if not hasattr(self, 'embeddings_'):
            raise ValueError('You should call .fit first!')

        # retrieve user factors
        u = torch.LongTensor(u).to(self.device)
        vu = self.embeddings_['user'](u)

        if self.training:
            # update item z
            vi = self.embeddings_['item'].weight

            if 'feat_item' in self.embeddings_:
                vfi = self.embeddings_['feat_item'].weight
                zfi = X['item'] @ vfi
                zfi2 = X['item']**2 @ vfi**2

                zi = (vi + zfi)[None]  # (None, n_items, k+1)
                zi2 = (vi**2 + zfi2)[None]
            else:
                zi, zi2 = vi[None], (vi**2)[None]

            if 'feat_user' in self.embeddings_:
                vfu = self.embeddings_['feat_user'].weight
                zfu = X['user'][u] @ vfu
                zfu2 = X['user'][u]**2 @ vfu**2

                zu = (vu + zfu)[:, None]
                zu2 = (vu**2 + zfu2)[:, None]
            else:
                zu, zu2 = vu[:, None], (vu**2)[:, None]

        else:
            zi, zi2 = self.zi[None], self.zi2[None]
            if 'feat_user' in self.embeddings_:
                zu, zu2 = self.zu[u][:, None], self.zu2[u][:, None]
            else:
                zu, zu2 = vu[:, None], (vu**2)[:, None]

        # forward & compute loss
        w = zi[..., -1] + zu[..., -1]
        v = (zi[..., :-1] + zu[..., :-1])**2
        v -= (zi2[..., :-1] + zu2[..., :-1])
        v = v.sum(-1) * .5
        s = self.w0 + w + v
        return s, [vu]

    def _compute_loss(self, s, y, c=None, loss='sgns',
                      aggregate="mean", alpha=1., weights=None):
        """
        s (torch.Tensor): scores
        y (torch.Tensor): target binary indicator (either {0, 1} or {-1, 1})
        c (torch.Tensor): confidence wrt to the y (optional)
        loss (str): type of the loss
        aggregate (str): aggregation method {'mean', 'sum', 'batchmean'}
        alpha (float): 1.
        weights (list of torch.Parameter):
                    weight whose l2 contraint to be explicitly computed
        """
        if loss in {'sgns', 'bpr'}:
            s = s.reshape(-1, self.n_negs + 1)
            y = y.reshape(-1, self.n_negs + 1)
            if loss == 'sgns':
                # requires y \in {-1, 1}
                l = -F.logsigmoid(s * y)

            elif loss == 'bpr':
                # assuming the first columns of scores are the postive
                # and the second is the negative
                # check the shape (should by (batch_sz, 2))
                if s.ndim != 2 or s.shape[-1] != 2:
                    raise ValueError('Should be 2 dimentional and only' +
                        'sample 1 negative sample')

                l = -F.logsigmoid((s * y).sum(1))[:, None]

            else:
                raise ValueError('[ERROR] only support {"sgns", "bpr", "cce", "kl"}')

            if aggregate == 'mean':
                l = l.mean()
            elif aggregate == 'sum':
                l = l.sum()
            elif aggregate == 'batchmean':
                l = l.sum(1).mean(0)

        else:
            # prepare the confidence matrix
            if c is not None:
                c = c.to_dense()
                c = c * alpha + 1

            y = y.to_dense()
            if loss == 'bce':
                if aggregate == 'batchmean':
                    aggregate = 'mean'

                l = F.binary_cross_entropy_with_logits(
                    s, y, reduction=aggregate, weight=c
                )

            elif loss == 'kl':
                l = F.kl_div(
                    torch.log_softmax(s, dim=-1), y / y.sum(1)[:, None],
                    reduction=aggregate
                )

            else:
                raise ValueError('[ERROR] only support {"sgns", "bpr", "cce", "kl"}')

        if (weights is not None) and (len(weights) > 0) and (self.l2 > 0):
            # adding L2 regularization term for sparse entries (user/item)
            l += self.l2 * sum([torch.sum(w**2) for w in weights])

        return l

    def fit(self, user_item, user_feature=None, item_feature=None,
            valid_user_item=None, feature_sparse=False, batch_sz=512,
            n_tests=1000, topk=100, valid_callback=None, verbose=False,
            keep_best=False, n_jobs=4):
        """"""
        # set some variables
        feats = {}
        do_valid = valid_user_item is not None
        n_users, n_items = user_item.shape
        desc_tmp = '[tloss=0.0000]'

        if not self.warm_start:
            self._init_embeddings(user_item, user_feature, item_feature)

        if self.target_device != 'cpu':
            self.to(self.target_device)

        if user_feature is not None:
            feats['user'] = torch.Tensor(user_feature).to(self.device)
        if item_feature is not None:
            feats['item'] = torch.Tensor(item_feature).to(self.device)

        # init optimizer
        self._init_optimizer()

        # prepare dataset
        if self.full_batch:
            dl = np.arange(0, n_users, batch_sz)
        else:
            dl = DataLoader(
                RecFeatDataset(user_item, n_negs=self.n_negs,
                               item_feature=item_feature,
                               user_feature=user_feature),
                batch_sz, collate_fn=collate_triplets_with_feature,
                shuffle=True, num_workers=n_jobs, drop_last=True
            )

        # do the optimization
        c = None
        val_score = 0
        self.best_val = 0
        self.best_model = None
        try:
            for _ in range(self.n_iters):
                with tqdm(total=len(dl), desc=desc_tmp,
                          ncols=80, disable=not verbose) as prog:

                    iter_loss = 0
                    val_timer = datetime.datetime.now()
                    rnd_usr = np.random.permutation(n_users)
                    for sample in dl:
                        # parse the output
                        if self.full_batch:
                            u = rnd_usr[sample:sample + batch_sz]
                            y = user_item[u].copy().tocoo()

                            # for loss weight
                            if (self.alpha is not None) and (self.alpha > 0):
                                c = y.copy()
                                c = scisp2tchsp(c).to(self.device)

                            y.data[:] = 1
                            y = scisp2tchsp(y).to(self.device)

                        else:
                            sample = tuple(d.to(self.device) for d in sample)
                            u, i, y, xu, xi = sample

                        # flush grads
                        self.opt.zero_grad()

                        # forward the model
                        if self.loss in {'bce', 'kl'}:
                            s, w = self.forward_batch(u, feats)
                        elif self.loss in {'sgns', 'bpr'}:
                            s, w = self.forward(u, i, {'user':xu, 'item':xi})

                        # compute the main loss
                        loss = self._compute_loss(
                            s, y, c, self.loss,
                            aggregate=self.loss_agg, weights=w
                        )

                        # backward
                        loss.backward()

                        # update params
                        self.opt.step()

                        # update loss and counter
                        iter_loss += loss.item()
                        now = datetime.datetime.now()
                        elapsed = now - val_timer
                        if do_valid and (elapsed.total_seconds() > 60):
                            val_timer = now
                            val_score = self.validate(
                                user_item, valid_user_item, feats,
                                n_tests, topk, valid_callback
                            )
                        prog.set_description(
                            '[tloss={:.4f} / vacc={:.4f}]'
                            .format(loss.item(), val_score)
                        )
                        prog.update(1)

                    if do_valid:
                        val_score = self.validate(
                            user_item, valid_user_item, feats,
                            n_tests, topk, valid_callback
                        )
                    prog.set_description(
                        '[tloss={:.4f} / vacc={:.4f}]'
                        .format(iter_loss / len(dl), val_score)
                    )

        except KeyboardInterrupt as e:
            print('User stopped the training!...')
        finally:
            # update the cached factors for faster inference
            if do_valid and keep_best:
                self.load_state_dict(self.best_model)
            self._update_z(feats)

    def validate(self, user_item, valid_user_item, feats, n_tests, topk,
                 callback=None):
        """"""
        if callback is None:
            scores = self._validate(
                user_item, valid_user_item, feats,
                n_tests, topk
            )
        else:
            scores = callback(
                self, user_item, valid_user_item, feats,
                n_tests, topk
            )

        # keep the best morel
        val_score = np.mean(scores)
        if val_score > self.best_val:
            self.best_val = val_score
            self.best_model = self.state_dict()

        return val_score

    def _validate(self, user_item, valid_user_item, feats,
                  n_tests, topk):
        self._update_z(feats)
        self.eval()
        scores = []
        n_users, n_items = user_item.shape
        test_users = np.random.choice(n_users, n_tests, False)
        for u in test_users:
            s = self.predict_user(u)[0]
            true = valid_user_item[u]
            train = user_item[u]
            score = compute_ndcg(s, true, train, topk)
            if score is None: continue
            scores.append(score)
        self.train()
        return scores
