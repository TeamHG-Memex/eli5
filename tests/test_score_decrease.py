# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from sklearn.svm import SVR

from eli5.score_decrease import iter_shuffled, get_feature_importances


def assert_column_mean_unchanged(X, **kwargs):
    mean = X.mean(axis=0)
    for X_sh in iter_shuffled(X, **kwargs):
        assert np.allclose(mean, X_sh.mean(axis=0))


def test_iter_shuffled_mean_preserved():
    X = np.arange(10 * 4).reshape(10, 4)
    assert_column_mean_unchanged(X)
    assert_column_mean_unchanged(X, columns_to_shuffle=[0, 1])
    assert_column_mean_unchanged(X, pre_shuffle=True)
    assert_column_mean_unchanged(X, columns_to_shuffle=[1], pre_shuffle=True)


def test_iter_shuffled_columns():
    X = np.arange(10 * 5).reshape(10, 5)

    Xs = [X_sh.copy() for X_sh in
          iter_shuffled(X, columns_to_shuffle=[2, 3], random_state=42)]
    assert len(Xs) == 2

    def is_shuffled(X, X_sh, col):
        return not np.allclose(X[:, col], X_sh[:, col])

    for X_sh in Xs:
        assert not is_shuffled(X, X_sh, 0)
        assert not is_shuffled(X, X_sh, 1)
        assert not is_shuffled(X, X_sh, 4)

    assert is_shuffled(X, Xs[0], 2)
    assert is_shuffled(X, Xs[1], 3)


def test_get_feature_importances(boston_train):
    X, y, feat_names = boston_train
    svr = SVR().fit(X, y)
    importances = get_feature_importances(svr.score, X, y)
    importances = dict(zip(feat_names, importances))
    print(importances)
    assert importances['AGE'] > importances['NOX']
    assert importances['B'] > importances['CHAS']
