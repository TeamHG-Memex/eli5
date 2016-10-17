# -*- coding: utf-8 -*-
from __future__ import absolute_import
from singledispatch import singledispatch

from lightning import classification, regression

from eli5.sklearn import (
    explain_linear_classifier_weights,
    explain_linear_regressor_weights,
    explain_prediction_linear_classifier,
    explain_prediction_linear_regressor
)


@singledispatch
def explain_weights(clf, vec=None, top=20, target_names=None,
                    feature_names=None, coef_scale=None):
    """ Return an explanation of an estimator """
    return {
        "estimator": repr(clf),
        "description": "Error: estimator %r is not supported" % clf,
    }


@singledispatch
def explain_prediction(clf, doc, vec=None, top=20, target_names=None,
                       feature_names=None, vectorized=False, coef_scale=None):
    """ Return an explanation of an estimator """
    return {
        "estimator": repr(clf),
        "description": "Error: estimator %r is not supported" % clf,
    }


_CLASSIFIERS = [
    classification.AdaGradClassifier,
    classification.CDClassifier,
    classification.FistaClassifier,
    classification.LinearSVC,
    classification.SAGAClassifier,
    classification.SAGClassifier,
    classification.SDCAClassifier,
    classification.SGDClassifier,
    classification.SVRGClassifier,
]

_REGRESSORS = [
    regression.AdaGradRegressor,
    regression.CDRegressor,
    regression.FistaRegressor,
    regression.LinearSVR,
    regression.SAGARegressor,
    regression.SAGRegressor,
    regression.SDCARegressor,
    regression.SGDRegressor,
    regression.SVRGRegressor
]

for clf in _CLASSIFIERS:
    explain_weights.register(clf, explain_linear_classifier_weights)
    explain_prediction.register(clf, explain_prediction_linear_classifier)


for reg in _REGRESSORS:
    explain_weights.register(reg, explain_linear_regressor_weights)
    explain_prediction.register(clf, explain_prediction_linear_regressor)
