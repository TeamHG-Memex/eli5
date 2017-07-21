"""
A module for computing feature importances by measuring how score decreases
when a feature is not available.

A similar method is described in Breiman, "Random Forests", Machine Learning,
45(1), 5-32, 2001 (available online at
https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf) as
"mean decrease in accuracy".
"""
from __future__ import absolute_import

import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


def _iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,
                   random_state=None):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in columns_to_shuffle:
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res
        X_res[:, columns] = X[:, columns]


def get_feature_importances(score_func, X, y, columns_to_shuffle=None,
                            random_state=None):
    score_base = score_func(X, y)
    Xs = _iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    return np.array([
        score_base - score_func(X_shuffled, y)
        for X_shuffled in Xs
    ])
