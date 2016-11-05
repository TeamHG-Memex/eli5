# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

from eli5.lime.utils import fit_proba


def test_fit_proba():
    X = np.array([
        [0.0, 0.8],
        [0.0, 0.5],
        [1.0, 0.1],
        [0.9, 0.2],
        [0.7, 0.3],
    ])
    y_proba = np.array([
        [0.0, 1.0],
        [0.1, 0.9],
        [1.0, 0.0],
        [0.55, 0.45],
        [0.4, 0.6],
    ])
    y_bin = y_proba.argmax(axis=1)

    # fit on binary labels
    clf = LogisticRegression(C=10, random_state=42)
    clf.fit(X, y_bin)
    y_pred = clf.predict_proba(X)[:,1]
    mae = mean_absolute_error(y_proba[:,1], y_pred)
    print(y_pred, mae)

    # fit on probabilities
    clf2 = LogisticRegression(C=10, random_state=42)
    fit_proba(clf2, X, y_proba, expand_factor=200, random_state=42)
    y_pred2 = clf2.predict_proba(X)[:,1]
    mae2 = mean_absolute_error(y_proba[:,1], y_pred2)
    print(y_pred2, mae2)

    assert mae2 * 1.2 < mae

    # let's get 3th example really right
    sample_weight = np.array([0.1, 0.1, 0.1, 10.0, 0.1])
    clf3 = LogisticRegression(C=10, random_state=42)
    fit_proba(clf3, X, y_proba, expand_factor=200, sample_weight=sample_weight,
              random_state=42)
    y_pred3 = clf3.predict_proba(X)[:,1]
    print(y_pred3)

    val = y_proba[3][1]
    assert abs(y_pred3[3] - val) * 1.5 < abs(y_pred2[3] - val)
    assert abs(y_pred3[3] - val) * 1.5 < abs(y_pred[3] - val)

    # without expand_factor it is just clf.fit
    clf4 = LogisticRegression(C=10, random_state=42)
    fit_proba(clf4, X, y_proba, expand_factor=None,
              random_state=42)
    y_pred4 = clf4.predict_proba(X)[:,1]
    assert np.allclose(y_pred, y_pred4)
