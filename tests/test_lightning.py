# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
pytest.importorskip('lightning')

from lightning import regression
from lightning.impl.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier

from eli5.lightning import _CLASSIFIERS, _REGRESSORS
from eli5 import explain_weights, explain_prediction
from .test_sklearn_explain_prediction import (
    assert_multiclass_linear_classifier_explained,
    assert_linear_regression_explained,
)
from .test_sklearn_explain_weights import (
    assert_explained_weights_linear_classifier,
    assert_explained_weights_linear_regressor,
)


@pytest.mark.parametrize(['clf'], [[clf()] for clf in _CLASSIFIERS])
def test_explain_predition_classifiers(newsgroups_train, clf):
    clf = OneVsRestClassifier(clf)
    assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction)


@pytest.mark.parametrize(['clf'], [[clf()] for clf in _CLASSIFIERS])
def test_explain_weights_classifiers(newsgroups_train, clf):
    clf = OneVsRestClassifier(clf)
    assert_explained_weights_linear_classifier(newsgroups_train, clf,
                                               add_bias=True)


regressors = [r for r in _REGRESSORS if r is not regression.SGDRegressor]
regressor_params = (
    [[reg()] for reg in regressors] +
    [[regression.SGDRegressor(max_iter=10, alpha=1e5, random_state=42)]]
)

@pytest.mark.parametrize(['reg'], regressor_params)
def test_explain_prediction_regressors(boston_train, reg):
    assert_linear_regression_explained(boston_train, reg, explain_prediction,
                                       atol=1e-3)


@pytest.mark.parametrize(['reg'], regressor_params)
def test_explain_weights_regressors(boston_train, reg):
    assert_explained_weights_linear_regressor(boston_train, reg,
                                              has_bias=False)


def test_explain_weights_unsupported():
    clf = BaseEstimator()
    res = explain_weights(clf)
    assert 'BaseEstimator' in res.error
    with pytest.raises(TypeError):
        explain_prediction(clf, unknown_argument=True)


def test_explain_prediction_unsupported():
    clf = BaseEstimator()
    doc = 'doc'
    res = explain_prediction(clf, doc)
    assert 'BaseEstimator' in res.error
    with pytest.raises(TypeError):
        explain_prediction(clf, doc, unknown_argument=True)
