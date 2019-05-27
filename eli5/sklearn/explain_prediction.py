# -*- coding: utf-8 -*-
from functools import partial

import numpy as np  # type: ignore
import scipy.sparse as sp  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.ensemble import (  # type: ignore
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (  # type: ignore
    ElasticNet,  # includes Lasso, MultiTaskElasticNet, etc.
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    Ridge,
    RidgeCV,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.svm import (  # type: ignore
    LinearSVC,
    LinearSVR,
    SVC,
    SVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.tree import (   # type: ignore
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from eli5.base import Explanation, TargetExplanation
from eli5.base_utils import singledispatch
from eli5.utils import (
    get_target_display_names,
    get_binary_target_scale_label_id
)
from eli5.sklearn.utils import (
    add_intercept,
    get_coef,
    get_default_target_names,
    get_X,
    get_X0,
    is_multiclass_classifier,
    is_multitarget_regressor,
    predict_proba,
    has_intercept,
    handle_vec,
)
from eli5.sklearn.text import add_weighted_spans
from eli5.explain import explain_prediction
from eli5._decision_path import DECISION_PATHS_CAVEATS
from eli5._feature_weights import get_top_features_filtered


@singledispatch
def explain_prediction_sklearn(estimator, doc,
                               vec=None,
                               top=None,
                               top_targets=None,
                               target_names=None,
                               targets=None,
                               feature_names=None,
                               feature_re=None,
                               feature_filter=None,
                               vectorized=False):
    """ Return an explanation of a scikit-learn estimator """
    return explain_prediction_sklearn_not_supported(estimator, doc)


@explain_prediction.register(BaseEstimator)
def explain_prediction_sklearn_not_supported(
        estimator, doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,
        feature_filter=None,
        vectorized=False):
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


def register(cls):
    def deco(f):
        return explain_prediction.register(cls)(
            explain_prediction_sklearn.register(cls)(f))
    return deco


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


@register(LogisticRegression)
@register(LogisticRegressionCV)
@register(SGDClassifier)
@register(PassiveAggressiveClassifier)
@register(Perceptron)
@register(LinearSVC)
@register(RidgeClassifier)
@register(RidgeClassifierCV)
def explain_prediction_linear_classifier(clf, doc,
                                         vec=None,
                                         top=None,
                                         top_targets=None,
                                         target_names=None,
                                         targets=None,
                                         feature_names=None,
                                         feature_re=None,
                                         feature_filter=None,
                                         vectorized=False,
                                         ):
    """
    Explain prediction of a linear classifier.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the classifier ``clf``
    (e.g. a fitted CountVectorizer instance); you can pass it
    instead of ``feature_names``.

    ``vectorized`` is a flag which tells eli5 if ``doc`` should be
    passed through ``vec`` or not. By default it is False, meaning that
    if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
    classifier. Set it to True if you're passing ``vec``, but ``doc``
    is already vectorized.
    """
    vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)

    proba = predict_proba(clf, X)
    score, = clf.decision_function(X)

    if has_intercept(clf):
        X = add_intercept(X)
    x = get_X0(X)

    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re, x)

    res = Explanation(
        estimator=repr(clf),
        method='linear model',
        targets=[],
    )
    assert res.targets is not None

    _weights = _linear_weights(clf, x, top, feature_names, flt_indices)
    classes = getattr(clf, "classes_", ["-1", "1"])  # OneClassSVM support
    display_names = get_target_display_names(classes, target_names,
                                             targets, top_targets, score)

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
        if len(display_names) == 1:  # target is passed explicitly
            label_id, target = display_names[0]
        else:
            label_id = 1 if score >= 0 else 0
            target = display_names[label_id][1]
        scale = -1 if label_id == 0 else 1

        target_expl = TargetExplanation(
            target=target,
            feature_weights=_weights(0, scale=scale),
            score=score,
            proba=proba[label_id] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


@register(NuSVC)
@register(SVC)
@register(OneClassSVM)
def test_explain_prediction_libsvm_linear(clf, doc, *args, **kwargs):
    if clf.kernel != 'linear':
        return Explanation(
            estimator=repr(clf),
            error="only kernel='linear' is currently supported for "
                  "libsvm-based classifiers",
        )
    if len(getattr(clf, 'classes_', [])) > 2:
        return Explanation(
            estimator=repr(clf),
            error="only binary libsvm-based classifiers are supported",
        )
    return explain_prediction_linear_classifier(clf, doc, *args, **kwargs)


@register(ElasticNet)
@register(ElasticNetCV)
@register(HuberRegressor)
@register(Lars)
@register(LassoCV)
@register(LinearRegression)
@register(LinearSVR)
@register(OrthogonalMatchingPursuit)
@register(OrthogonalMatchingPursuitCV)
@register(PassiveAggressiveRegressor)
@register(Ridge)
@register(RidgeCV)
@register(SGDRegressor)
@register(TheilSenRegressor)
@register(SVR)
@register(NuSVR)
def explain_prediction_linear_regressor(reg, doc,
                                        vec=None,
                                        top=None,
                                        top_targets=None,
                                        target_names=None,
                                        targets=None,
                                        feature_names=None,
                                        feature_re=None,
                                        feature_filter=None,
                                        vectorized=False):
    """
    Explain prediction of a linear regressor.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the classifier ``clf``;
    you can pass it instead of ``feature_names``.

    ``vectorized`` is a flag which tells eli5 if ``doc`` should be
    passed through ``vec`` or not. By default it is False, meaning that
    if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
    regressor ``reg``. Set it to True if you're passing ``vec``,
    but ``doc`` is already vectorized.
    """
    if isinstance(reg, (SVR, NuSVR)) and reg.kernel != 'linear':
        return explain_prediction_sklearn_not_supported(reg, doc)

    vec, feature_names = handle_vec(reg, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)

    score, = reg.predict(X)

    if has_intercept(reg):
        X = add_intercept(X)
    x = get_X0(X)

    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re, x)

    res = Explanation(
        estimator=repr(reg),
        method='linear model',
        targets=[],
        is_regression=True,
    )
    assert res.targets is not None

    _weights = _linear_weights(reg, x, top, feature_names, flt_indices)
    names = get_default_target_names(reg)
    display_names = get_target_display_names(names, target_names, targets,
                                             top_targets, score)

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


DECISION_PATHS_CAVEATS = """
Feature weights are calculated by following decision paths in trees
of an ensemble (or a single tree for DecisionTreeClassifier).
Each node of the tree has an output score, and contribution of a feature
on the decision path is how much the score changes from parent to child.
Weights of all features sum to the output score or proba of the estimator.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_TREE_CLF_BINARY = """
Features with largest coefficients.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_TREE_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_TREE_REG = """
Features with largest coefficients.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_TREE_REG_MULTITARGET = """
Features with largest coefficients per target.
""" + DECISION_PATHS_CAVEATS


@register(DecisionTreeClassifier)
@register(ExtraTreesClassifier)
@register(GradientBoostingClassifier)
@register(RandomForestClassifier)
def explain_prediction_tree_classifier(
        clf, doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,
        feature_filter=None,
        vectorized=False):
    """ Explain prediction of a tree classifier.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the classifier ``clf``
    (e.g. a fitted CountVectorizer instance); you can pass it
    instead of ``feature_names``.

    ``vectorized`` is a flag which tells eli5 if ``doc`` should be
    passed through ``vec`` or not. By default it is False, meaning that
    if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
    classifier. Set it to True if you're passing ``vec``,
    but ``doc`` is already vectorized.

    Method for determining feature importances follows an idea from
    http://blog.datadive.net/interpreting-random-forests/.
    Feature weights are calculated by following decision paths in trees
    of an ensemble (or a single tree for DecisionTreeClassifier).
    Each node of the tree has an output score, and contribution of a feature
    on the decision path is how much the score changes from parent to child.
    Weights of all features sum to the output score or proba of the estimator.
    """
    vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized)
    if feature_names.bias_name is None:
        # Tree estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    proba = predict_proba(clf, X)
    if hasattr(clf, 'decision_function'):
        score, = clf.decision_function(X)
    else:
        score = None

    is_multiclass = clf.n_classes_ > 2
    feature_weights = _trees_feature_weights(
        clf, X, feature_names, clf.n_classes_)
    x = get_X0(add_intercept(X))
    flt_feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re, x)

    def _weights(label_id, scale=1.0):
        weights = feature_weights[:, label_id]
        return get_top_features_filtered(x, flt_feature_names, flt_indices,
                                         weights, top, scale)

    res = Explanation(
        estimator=repr(clf),
        method='decision path',
        targets=[],
        description=(DESCRIPTION_TREE_CLF_MULTICLASS if is_multiclass
                     else DESCRIPTION_TREE_CLF_BINARY),
    )
    assert res.targets is not None

    display_names = get_target_display_names(
        clf.classes_, target_names, targets, top_targets,
        score=score if score is not None else proba)

    if is_multiclass:
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
        target, scale, label_id = get_binary_target_scale_label_id(
            score, display_names, proba)
        target_expl = TargetExplanation(
            target=target,
            feature_weights=_weights(label_id, scale=scale),
            score=score if score is not None else None,
            proba=proba[label_id] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


@register(DecisionTreeRegressor)
@register(ExtraTreesRegressor)
@register(GradientBoostingRegressor)
@register(RandomForestRegressor)
def explain_prediction_tree_regressor(
        reg, doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,
        feature_filter=None,
        vectorized=False):
    """ Explain prediction of a tree regressor.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the regressor ``reg``
    (e.g. a fitted CountVectorizer instance); you can pass it
    instead of ``feature_names``.

    ``vectorized`` is a flag which tells eli5 if ``doc`` should be
    passed through ``vec`` or not. By default it is False, meaning that
    if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
    regressor. Set it to True if you're passing ``vec``,
    but ``doc`` is already vectorized.

    Method for determining feature importances follows an idea from
    http://blog.datadive.net/interpreting-random-forests/.
    Feature weights are calculated by following decision paths in trees
    of an ensemble (or a single tree for DecisionTreeRegressor).
    Each node of the tree has an output score, and contribution of a feature
    on the decision path is how much the score changes from parent to child.
    Weights of all features sum to the output score of the estimator.
    """
    vec, feature_names = handle_vec(reg, doc, vec, vectorized, feature_names)
    X = get_X(doc, vec=vec, vectorized=vectorized)
    if feature_names.bias_name is None:
        # Tree estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    score, = reg.predict(X)
    num_targets = getattr(reg, 'n_outputs_', 1)
    is_multitarget = num_targets > 1
    feature_weights = _trees_feature_weights(reg, X, feature_names, num_targets)
    x = get_X0(add_intercept(X))
    flt_feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re, x)

    def _weights(label_id, scale=1.0):
        weights = feature_weights[:, label_id]
        return get_top_features_filtered(x, flt_feature_names, flt_indices,
                                         weights, top, scale)

    res = Explanation(
        estimator=repr(reg),
        method='decision path',
        description=(DESCRIPTION_TREE_REG_MULTITARGET if is_multitarget
                     else DESCRIPTION_TREE_REG),
        targets=[],
        is_regression=True,
    )
    assert res.targets is not None

    names = get_default_target_names(reg, num_targets=num_targets)
    display_names = get_target_display_names(names, target_names, targets,
                                             top_targets, score)

    if is_multitarget:
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


def _trees_feature_weights(clf, X, feature_names, num_targets):
    """ Return feature weights for a tree or a tree ensemble.
    """
    feature_weights = np.zeros([len(feature_names), num_targets])
    is_grad_boost = isinstance(clf, (GradientBoostingClassifier,
                                     GradientBoostingRegressor))
    if hasattr(clf, 'tree_'):
        _update_tree_feature_weights(X, feature_names, clf, feature_weights)
    else:
        if is_grad_boost:
            weight = clf.learning_rate
        else:
            weight = 1. / len(clf.estimators_)
        for _clfs in clf.estimators_:
            _update = partial(_update_tree_feature_weights, X, feature_names)
            if isinstance(_clfs, np.ndarray):
                if len(_clfs) == 1:
                    _update(_clfs[0], feature_weights)
                else:
                    for idx, _clf in enumerate(_clfs):
                        _update(_clf, feature_weights[:, idx])
            else:
                _update(_clfs, feature_weights)
        feature_weights *= weight
        if hasattr(clf, 'init_'):
            if clf.init_ == 'zero':
                bias_init = 0
            elif is_grad_boost and hasattr(clf.loss_, 'get_init_raw_predictions'):
                bias_init = clf.loss_.get_init_raw_predictions(
                    X, clf.init_).astype(np.float64)[0]
            else:
                bias_init = clf.init_.predict(X)[0]
            feature_weights[feature_names.bias_idx] += bias_init
    return feature_weights


def _update_tree_feature_weights(X, feature_names, clf, feature_weights):
    """ Update tree feature weights using decision path method.
    """
    tree_value = clf.tree_.value
    if tree_value.shape[1] == 1:
        squeeze_axis = 1
    else:
        assert tree_value.shape[2] == 1
        squeeze_axis = 2
    tree_value = np.squeeze(tree_value, axis=squeeze_axis)
    tree_feature = clf.tree_.feature
    _, indices = clf.decision_path(X).nonzero()
    if isinstance(clf, DecisionTreeClassifier):
        norm = lambda x: x / x.sum()
    else:
        norm = lambda x: x
    feature_weights[feature_names.bias_idx] += norm(tree_value[0])
    for parent_idx, child_idx in zip(indices, indices[1:]):
        assert tree_feature[parent_idx] >= 0
        feature_idx = tree_feature[parent_idx]
        diff = norm(tree_value[child_idx]) - norm(tree_value[parent_idx])
        feature_weights[feature_idx] += diff



def _multiply(X, coef):
    """ Multiple X by coef element-wise, preserving sparsity. """
    if sp.issparse(X):
        return X.multiply(sp.csr_matrix(coef))
    else:
        return np.multiply(X, coef)


def _linear_weights(clf, x, top, flt_feature_names, flt_indices):
    """ Return top weights getter for label_id.
    """
    def _weights(label_id, scale=1.0):
        coef = get_coef(clf, label_id)
        scores = _multiply(x, coef)
        return get_top_features_filtered(x, flt_feature_names, flt_indices,
                                         scores, top, scale)
    return _weights
