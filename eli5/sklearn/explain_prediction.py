# -*- coding: utf-8 -*-
from singledispatch import singledispatch
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
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
from sklearn.tree import DecisionTreeClassifier

from eli5.base import Explanation, TargetExplanation
from eli5.utils import get_target_display_names
from eli5.sklearn.utils import (
    get_coef,
    get_default_target_names,
    get_X,
    is_multiclass_classifier,
    is_multitarget_regressor,
    predict_proba,
    has_intercept,
    handle_vec,
)
from eli5.sklearn.text import add_weighted_spans
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
    vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)

    proba = predict_proba(clf, X)
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

    display_names = get_target_display_names(clf.classes_, target_names,
                                             targets)

    if is_multiclass_classifier(clf):
        for label_id, label in display_names:
            target_expl = TargetExplanation(
                target=label,
                feature_weights=_weights(label_id),
                score=score[label_id],
                proba=proba[label_id] if proba is not None else None,
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            res.targets.append(target_expl)
    else:
        target_expl = TargetExplanation(
            target=display_names[1][1],
            feature_weights=_weights(0),
            score=score,
            proba=proba[1] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


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
    vec, feature_names = handle_vec(reg, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)

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
    display_names = get_target_display_names(names, target_names, targets)

    if is_multitarget_regressor(reg):
        for label_id, label in display_names:
            target_expl = TargetExplanation(
                target=label,
                feature_weights=_weights(label_id),
                score=score[label_id],
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            res.targets.append(target_expl)
    else:
        target_expl = TargetExplanation(
            target=display_names[0][1],
            feature_weights=_weights(0),
            score=score,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


@explain_prediction_sklearn.register(DecisionTreeClassifier)
@explain_prediction_sklearn.register(GradientBoostingClassifier)
@explain_prediction_sklearn.register(AdaBoostClassifier)
@explain_prediction_sklearn.register(RandomForestClassifier)
@explain_prediction_sklearn.register(ExtraTreesClassifier)
def explain_prediction_tree_classifier(
        clf, doc,
        vec=None,
        top=None,
        target_names=None,
        targets=None,
        feature_names=None,
        vectorized=False):
    """ Explain prediction of a tree classifier. """
    vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized)
    if feature_names.bias_name is None:
        # XGBoost estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    proba = predict_proba(clf, X)
    if hasattr(clf, 'decision_function'):
        score, = clf.decision_function(X)
    else:
        score = None

    feature_weights = np.zeros([len(feature_names), clf.n_classes_])
    if hasattr(clf, 'tree_'):
        _update_tree_weights(clf, X, feature_names, feature_weights)
    else:
        # Possible optimization: use clf.decision_path
        for _clf in clf.estimators_:
            if isinstance(_clf, np.ndarray):
                assert len(_clf) == 1
                _clf = _clf[0]
            _update_tree_weights(_clf, X, feature_names, feature_weights)

    def _weights(label_id):
        scores = feature_weights[:, label_id]
        return get_top_features(feature_names, scores, top)

    res = Explanation(
        estimator=repr(clf),
        method='decision path',
        targets=[],
    )

    display_names = get_target_display_names(
        clf.classes_, target_names, targets)

    if clf.n_classes_ > 2:
        for label_id, label in display_names:
            target_expl = TargetExplanation(
                target=label,
                feature_weights=_weights(label_id),
                score=score[label_id] if score is not None else None,
                proba=proba[label_id] if proba is not None else None,
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            res.targets.append(target_expl)
    else:
        target_expl = TargetExplanation(
            target=display_names[1][1],
            feature_weights=_weights(1),
            score=score if score is not None else None,
            proba=proba[1] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


def _update_tree_weights(clf, X, feature_names, feature_weights):
    tree_value = clf.tree_.value
    assert tree_value.shape[1] == 1
    tree_feature = clf.tree_.feature
    _, indices = clf.decision_path(X).nonzero()
    feature_weights[feature_names.bias_idx] += tree_value[0, 0]
    for parent_idx, child_idx in zip(indices, indices[1:]):
        feature_weights[tree_feature[parent_idx]] += (
            tree_value[child_idx, 0] - tree_value[parent_idx, 0])


def _multiply(X, coef):
    """ Multiple X by coef element-wise, preserving sparsity. """
    if sp.issparse(X):
        return X.multiply(sp.csr_matrix(coef))
    else:
        return np.multiply(X, coef)


def _add_intercept(X):
    """ Add intercept column to X """
    intercept = np.ones((X.shape[0], 1))
    if sp.issparse(X):
        return sp.hstack([X, intercept]).tocsr()
    else:
        return np.hstack([X, intercept])
