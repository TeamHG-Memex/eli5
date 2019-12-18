# -*- coding: utf-8 -*-
from __future__ import absolute_import

from lightning.impl.base import BaseEstimator
from lightning import classification, regression
from sklearn.multiclass import OneVsRestClassifier

from eli5.base import Explanation
from eli5.base_utils import singledispatch
from eli5.sklearn import (
    explain_linear_classifier_weights,
    explain_linear_regressor_weights,
    explain_prediction_linear_classifier,
    explain_prediction_linear_regressor
)
from eli5.explain import explain_prediction, explain_weights


@singledispatch
def explain_weights_lightning(estimator, vec=None, top=20, target_names=None,
                              targets=None, feature_names=None,
                              coef_scale=None):
    """ Return an explanation of a lightning estimator weights """
    return explain_weights_lightning_not_supported(estimator)


@explain_weights.register(BaseEstimator)
def explain_weights_lightning_not_supported(
        estimator, vec=None, top=20, target_names=None,
        targets=None, feature_names=None,
        coef_scale=None):
    return Explanation(
        estimator=repr(estimator),
        error="Error: estimator %r is not supported" % estimator,
    )


@singledispatch
def explain_prediction_lightning(estimator, doc, vec=None, top=None,
                                 target_names=None, targets=None,
                                 feature_names=None, vectorized=False,
                                 coef_scale=None):
    """ Return an explanation of a lightning estimator predictions """
    return explain_weights_lightning_not_supported(estimator, doc)


@explain_prediction.register(BaseEstimator)
def explain_prediction_lightning_not_supported(
        estimator, doc, vec=None, top=None,
        target_names=None, targets=None,
        feature_names=None, vectorized=False,
        coef_scale=None):
    return Explanation(
        estimator=repr(estimator),
        error="Error: estimator %r is not supported" % estimator,
    )


@explain_prediction_lightning.register(OneVsRestClassifier)
def explain_prediction_ovr_lightning(clf, doc, **kwargs):
    # dispatch OvR to eli5.lightning
    # if explain_prediction_lightning is called explicitly
    estimator = clf.estimator
    func = explain_prediction_lightning.dispatch(estimator.__class__)
    return func(clf, doc, **kwargs)


@explain_weights_lightning.register(OneVsRestClassifier)
def explain_weights_ovr_lightning(ovr, **kwargs):
    # dispatch OvR to eli5.lightning
    # if explain_weights_lightning is called explicitly
    estimator = ovr.estimator
    func = explain_weights_lightning.dispatch(estimator.__class__)
    return func(ovr, **kwargs)


_CLASSIFIERS = [
    classification.AdaGradClassifier,
    classification.CDClassifier,
    classification.FistaClassifier,
    classification.LinearSVC,
    classification.SAGAClassifier,
    classification.SAGClassifier,
    classification.SDCAClassifier,
    classification.SGDClassifier,
    # classification.SVRGClassifier,   # tests fail for it
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
    # regression.SVRGRegressor
]

for clf in _CLASSIFIERS:
    explain_weights.register(clf, explain_linear_classifier_weights)
    explain_weights_lightning.register(clf, explain_linear_classifier_weights)
    explain_prediction.register(clf, explain_prediction_linear_classifier)
    explain_prediction_lightning.register(clf, explain_prediction_linear_classifier)


for reg in _REGRESSORS:
    explain_weights.register(reg, explain_linear_regressor_weights)
    explain_weights_lightning.register(reg, explain_linear_regressor_weights)
    explain_prediction.register(reg, explain_prediction_linear_regressor)
    explain_prediction_lightning.register(reg, explain_prediction_linear_regressor)
