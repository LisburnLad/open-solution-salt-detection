import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator

class KFoldBySortedValue(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits