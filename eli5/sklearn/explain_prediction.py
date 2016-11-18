# -*- coding: utf-8 -*-
from singledispatch import singledispatch

import numpy as np
import scipy.sparse as sp
import six
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import (
    ElasticNet,  # includes Lasso, MultiTaskElasticNet, etc.
    ElasticNetCV,
    Lars,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    Ridge,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier

from eli5.base import Explanation, TargetExplanation
from eli5.utils import get_display_names
from eli5.sklearn.unhashing import InvertableHashingVectorizer, is_invhashing
from eli5.sklearn.utils import (
    get_feature_names,
    get_coef,
    get_default_target_names,
    is_multiclass_classifier,
    is_multitarget_regressor,
    is_probabilistic_classifier,
    has_intercept,
)
from eli5.sklearn.text import get_weighted_spans
from eli5._feature_weights import get_top_features
from eli5.explain import explain_prediction


@explain_prediction.register(BaseEstimator)
@singledispatch
def explain_prediction_sklearn(estimator, doc,
                               vec=None,
                               top=None,
                               target_names=None,
                               targets=None,
                               feature_names=None,
                               vectorized=False):
    """ Return an explanation of a scikit-learn estimator """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


@explain_prediction.register(OneVsRestClassifier)
def explain_prediction_ovr(clf, doc, **kwargs):
    estimator = clf.estimator
    func = explain_prediction.dispatch(estimator.__class__)
    return func(clf, doc, **kwargs)


@explain_prediction_sklearn.register(OneVsRestClassifier)
def explain_prediction_ovr_sklearn(clf, doc, **kwargs):
    # dispatch OvR to eli5.sklearn
    # if explain_prediction_sklearn is called explicitly
    estimator = clf.estimator
    func = explain_prediction_sklearn.dispatch(estimator.__class__)
    return func(clf, doc, **kwargs)


@explain_prediction_sklearn.register(LogisticRegression)
@explain_prediction_sklearn.register(LogisticRegressionCV)
@explain_prediction_sklearn.register(SGDClassifier)
@explain_prediction_sklearn.register(PassiveAggressiveClassifier)
@explain_prediction_sklearn.register(Perceptron)
@explain_prediction_sklearn.register(LinearSVC)
def explain_prediction_linear_classifier(clf, doc,
                                         vec=None,
                                         top=None,
                                         target_names=None,
                                         targets=None,
                                         feature_names=None,
                                         vectorized=False):
    """ Explain prediction of a linear classifier. """
    vec, feature_names = _handle_vec(clf, doc, vec, vectorized, feature_names)
    X = _get_X(doc, vec=vec, vectorized=vectorized)

    if is_probabilistic_classifier(clf):
        try:
            proba, = clf.predict_proba(X)
        except NotImplementedError:
            proba = None
    else:
        proba = None
    score, = clf.decision_function(X)

    if has_intercept(clf):
        X = _add_intercept(X)
    x, = X

    res = Explanation(
        estimator=repr(clf),
        method='linear model',
        targets=[],
    )

    def _weights(label_id):
        coef = get_coef(clf, label_id)
        scores = _multiply(x, coef)
        return get_top_features(feature_names, scores, top)

    display_names = get_display_names(clf.classes_, target_names, targets)

    if is_multiclass_classifier(clf):
        for label_id, label in display_names:
            target_expl = TargetExplanation(
                target=label,
                feature_weights=_weights(label_id),
                score=score[label_id],
                proba=proba[label_id] if proba is not None else None,
            )
            _add_weighted_spans(doc, vec, target_expl)
            res.targets.append(target_expl)
    else:
        target_expl = TargetExplanation(
            target=display_names[1][1],
            feature_weights=_weights(0),
            score=score,
            proba=proba[1] if proba is not None else None,
        )
        _add_weighted_spans(doc, vec, target_expl)
        res.targets.append(target_expl)

    return res


def _add_weighted_spans(doc, vec, target_expl):
    if vec is not None:
        weighted_spans = get_weighted_spans(
            doc, vec, target_expl.feature_weights)
        if weighted_spans:
            target_expl.weighted_spans = weighted_spans


def _multiply(X, coef):
    if sp.issparse(X):
        return X.multiply(sp.csr_matrix(coef))
    else:
        return np.multiply(X, coef)


def _add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    if sp.issparse(X):
        return sp.hstack([X, intercept]).tocsr()
    else:
        return np.hstack([X, intercept])


def _get_X(doc, vec=None, vectorized=False):
    if vec is None or vectorized:
        X = np.array([doc]) if isinstance(doc, np.ndarray) else doc
    else:
        X = vec.transform([doc])
    if sp.issparse(X):
        X = X.toarray()
    return X


def _handle_vec(clf, doc, vec, vectorized, feature_names):
    if isinstance(vec, HashingVectorizer) and not vectorized:
        vec = InvertableHashingVectorizer(vec)
        vec.fit([doc])
    if is_invhashing(vec) and feature_names is None:
        # Explaining predictions does not need coef_scale,
        # because it is handled by the vectorizer.
        feature_names = vec.get_feature_names(always_signed=False)
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)
    return vec, feature_names


@explain_prediction_sklearn.register(ElasticNet)
@explain_prediction_sklearn.register(ElasticNetCV)
@explain_prediction_sklearn.register(Lars)
@explain_prediction_sklearn.register(LinearRegression)
@explain_prediction_sklearn.register(LinearSVR)
@explain_prediction_sklearn.register(Ridge)
@explain_prediction_sklearn.register(RidgeCV)
@explain_prediction_sklearn.register(SGDRegressor)
def explain_prediction_linear_regressor(reg, doc,
                                        vec=None,
                                        top=None,
                                        target_names=None,
                                        targets=None,
                                        feature_names=None,
                                        vectorized=False):
    """ Explain prediction of a linear regressor. """
    vec, feature_names = _handle_vec(reg, doc, vec, vectorized, feature_names)
    X = _get_X(doc, vec=vec, vectorized=vectorized)

    score, = reg.predict(X)

    if has_intercept(reg):
        X = _add_intercept(X)
    x, = X

    res = Explanation(
        estimator=repr(reg),
        method='linear model',
        targets=[],
        is_regression=True,
    )

    def _weights(label_id):
        coef = get_coef(reg, label_id)
        scores = _multiply(x, coef)
        return get_top_features(feature_names, scores, top)

    names = get_default_target_names(reg)
    display_names = get_display_names(names, target_names, targets)

    if is_multitarget_regressor(reg):
        for label_id, label in display_names:
            target_expl = TargetExplanation(
                target=label,
                feature_weights=_weights(label_id),
                score=score[label_id],
            )
            _add_weighted_spans(doc, vec, target_expl)
            res.targets.append(target_expl)
    else:
        target_expl = TargetExplanation(
            target=display_names[0][1],
            feature_weights=_weights(0),
            score=score,
        )
        _add_weighted_spans(doc, vec, target_expl)
        res.targets.append(target_expl)

    return res
