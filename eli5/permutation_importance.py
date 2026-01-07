"""
A module for computing feature importances by measuring how score decreases
when a feature is not available. It contains basic building blocks;
there is a full-featured sklearn-compatible implementation
in :class:`~.PermutationImportance`.

A similar method is described in Breiman, "Random Forests", Machine Learning,
45(1), 5-32, 2001 (available online at
https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf), with an
application to random forests. It is known in literature as
"Mean Decrease Accuracy (MDA)" or "permutation importance".
"""
from __future__ import absolute_import
from typing import Tuple, List, Callable, Any

import numpy as np
from sklearn.utils import check_random_state


def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,
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


def get_score_importances(
        score_func,  # type: Callable[[Any, Any], float]
        X,
        y,
        n_iter=5,  # type: int
        columns_to_shuffle=None,
        random_state=None
    ):
    # type: (...) -> Tuple[float, List[np.ndarray]]
    """
    Return ``(base_score, score_decreases)`` tuple with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.

    If you just want feature importances, you can take a mean of the result::

        import numpy as np
        from eli5.permutation_importance import get_score_importances

        base_score, score_decreases = get_score_importances(score_func, X, y)
        feature_importances = np.mean(score_decreases, axis=0)

    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for i in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle,
            random_state=rng
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases


def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None,
                        random_state=None):
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])
