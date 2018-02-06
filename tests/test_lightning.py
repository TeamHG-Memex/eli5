# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
pytest.importorskip('lightning')

from lightning import regression, classification
from lightning.impl.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier

from eli5.lightning import _CLASSIFIERS, _REGRESSORS
from eli5 import (
    explain_weights, explain_weights_lightning,
    explain_prediction, explain_prediction_lightning,
)
from .test_sklearn_explain_prediction import (
    assert_multiclass_linear_classifier_explained,
    assert_linear_regression_explained,
    assert_binary_linear_classifier_explained,
    test_explain_prediction_pandas as _check_explain_prediction_pandas,
)
from .test_sklearn_explain_weights import (
    assert_explained_weights_linear_classifier,
    assert_explained_weights_linear_regressor,
)

_DEFAULTS = {
    classification.AdaGradClassifier: dict(alpha=1e-9, random_state=42),
    classification.SDCAClassifier: dict(alpha=1e-9, random_state=42),
    regression.SGDRegressor: dict(max_iter=10, alpha=1e5, random_state=42)
}

def _instances(classes):
    return [[cls(**_DEFAULTS.get(cls, {}))] for cls in classes]


@pytest.mark.parametrize(['clf'], _instances(_CLASSIFIERS))
def test_explain_predition_classifiers(newsgroups_train, clf):
    clf = OneVsRestClassifier(clf)
    assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction)
    if _CLASSIFIERS.index(type(clf.estimator)) == 0:
        assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                      explain_prediction_lightning)


@pytest.mark.parametrize(['clf'], _instances(_CLASSIFIERS))
def test_explain_predition_classifiers_binary(newsgroups_train_binary, clf):
    assert_binary_linear_classifier_explained(newsgroups_train_binary, clf,
                                              explain_prediction)


@pytest.mark.parametrize(['clf'], _instances(_CLASSIFIERS))
def test_explain_weights_classifiers(newsgroups_train, clf):
    clf = OneVsRestClassifier(clf)
    assert_explained_weights_linear_classifier(newsgroups_train, clf,
                                               add_bias=True)
    if _CLASSIFIERS.index(type(clf.estimator)) == 0:
        assert_explained_weights_linear_classifier(
            newsgroups_train, clf, add_bias=True,
            explain_weights=explain_weights_lightning)


@pytest.mark.parametrize(['reg'], _instances(_REGRESSORS))
def test_explain_prediction_regressors(boston_train, reg):
    assert_linear_regression_explained(boston_train, reg, explain_prediction,
                                       atol=1e-3)


@pytest.mark.parametrize(['reg'], _instances(_REGRESSORS))
def test_explain_weights_regressors(boston_train, reg):
    assert_explained_weights_linear_regressor(boston_train, reg,
                                              has_bias=False)


@pytest.mark.parametrize(['reg'], _instances(_REGRESSORS)[:2])
def test_explain_prediction_pandas(reg, boston_train):
    _check_explain_prediction_pandas(reg, boston_train)


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
