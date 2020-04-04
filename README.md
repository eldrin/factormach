# *FactorMach* : Yet Another Factorization Machine for RecSys

***FactorMach*** is yet another implementation of the Factorization Machine [^rendle10]. It aims for providing rich options for fitting FM models. Specifically

- allowing various objective functions

- gpu support

Currently such features are heavily dependent on [PyTorch](https://pytorch.org/).


## Getting Started

It's not on PyPI yet, means you need to install the package using the `pip` and `git`

```console
$ pip install git+https://github.com/eldrin/factormach.git@master
```

### Quick Look

Basic usage is similar to the scikit-learn

```python

import numpy as np
from factormach.models.uifm import UserItemFM
from factormach.utils import load_triplet_csv

# load interaction data
user_item = load_triplet_csv('/path/to/triplets.txt')
item_feat = np.load('/path/to/item_features.npy')

# initialize user-item fm
fm = UserItemFM(k=32, n_iters=20, loss='bce')

# train
fm.fit(
  user_item,
  item_feature=item_feat,
  batch_sz=512
)

```

## Current Status & Contributing

As an pre-alpha version, currently we mostly provide the API specialized on the recommendation. Although we plan to extend the API as general as possible in the near future. If interested, feel free to send pull requests and drop issues. We are more than happy to listen what do you think.


## Authors

- Jaehun Kim

## License


This project is licensed under the MIT License - see the LICENSE.md file for details


## Reference

[^rendle10] S. Rendle, "Factorization Machines," 2010 IEEE International Conference on Data Mining, Sydney, NSW, 2010, pp. 995-1000. 
