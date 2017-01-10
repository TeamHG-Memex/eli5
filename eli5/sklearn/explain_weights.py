# -*- coding: utf-8 -*-
from __future__ import absolute_import
from singledispatch import singledispatch

import numpy as np  # type: ignore

from sklearn.base import BaseEstimator  # type: ignore
from sklearn.linear_model import (   # type: ignore
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
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.svm import LinearSVC, LinearSVR  # type: ignore
# TODO: see https://github.com/scikit-learn/scikit-learn/pull/2250
from sklearn.naive_bayes import BernoulliNB, MultinomialNB    # type: ignore
from sklearn.ensemble import (  # type: ignore
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.tree import (  # type: ignore
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from eli5.base import (
    Explanation, TargetExplanation, FeatureWeight, FeatureImportances)
from eli5._feature_weights import get_top_features
from eli5.utils import argsort_k_largest_positive, get_target_display_names
from eli5.sklearn.unhashing import handle_hashing_vec, is_invhashing
from eli5.sklearn.treeinspect import get_tree_info
from eli5.sklearn.utils import (
    get_coef,
    is_multiclass_classifier,
    is_multitarget_regressor,
    get_feature_names,
    get_default_target_names,
)
from eli5.explain import explain_weights


LINEAR_CAVEATS = """
Caveats:
1. Be careful with features which are not
   independent - weights don't show their importance.
2. If scale of input features is different then scale of coefficients
   will also be different, making direct comparison between coefficient values
   incorrect.
3. Depending on regularization, rare features sometimes may have high
   coefficients; this doesn't mean they contribute much to the
   classification result for most examples.
""".lstrip()

HASHING_CAVEATS = """
Feature names are restored from their hashes; this is not 100% precise
because collisions are possible. For known collisions possible feature names
are separated by | sign. Keep in mind the collision list is not exhaustive.
Features marked with (-) should be read as inverted: if they have positive
coefficient, the result is negative, if they have negative coefficient,
the result is positive.
""".lstrip()

DESCRIPTION_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + LINEAR_CAVEATS

DESCRIPTION_CLF_BINARY = """
Features with largest coefficients.
""" + LINEAR_CAVEATS

DESCRIPTION_REGRESSION = DESCRIPTION_CLF_BINARY

DESCRIPTION_REGRESSION_MULTITARGET = """
Features with largest coefficients per target.
""" + LINEAR_CAVEATS

DESCRIPTION_RANDOM_FOREST = """
Random forest feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

DESCRIPTION_DECISION_TREE = """
Decision tree feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

_TOP = 20


@explain_weights.register(BaseEstimator)
@singledispatch
def explain_weights_sklearn(estimator, vec=None, top=_TOP,
                            target_names=None,
                            targets=None,
                            feature_names=None, coef_scale=None,
                            feature_re=None, feature_filter=None):
    """ Return an explanation of an estimator """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


@explain_weights.register(OneVsRestClassifier)
def explain_weights_ovr(ovr, **kwargs):
    estimator = ovr.estimator
    func = explain_weights.dispatch(estimator.__class__)
    return func(ovr, **kwargs)


@explain_weights_sklearn.register(OneVsRestClassifier)
def explain_weights_ovr_sklearn(ovr, **kwargs):
    # dispatch OvR to eli5.sklearn
    # if explain_prediction_sklearn is called explicitly
    estimator = ovr.estimator
    func = explain_weights_sklearn.dispatch(estimator.__class__)
    return func(ovr, **kwargs)


@explain_weights_sklearn.register(LogisticRegression)
@explain_weights_sklearn.register(LogisticRegressionCV)
@explain_weights_sklearn.register(SGDClassifier)
@explain_weights_sklearn.register(PassiveAggressiveClassifier)
@explain_weights_sklearn.register(Perceptron)
@explain_weights_sklearn.register(LinearSVC)
@explain_weights_sklearn.register(RidgeClassifier)
@explain_weights_sklearn.register(RidgeClassifierCV)
def explain_linear_classifier_weights(clf,
                                      vec=None,
                                      top=_TOP,
                                      target_names=None,
                                      targets=None,
                                      feature_names=None,
                                      coef_scale=None,
                                      feature_re=None,
                                      feature_filter=None,
                                      ):
    """
    Return an explanation of a linear classifier weights.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``target_names``, ``targets``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the classifier ``clf``
    (e.g. a fitted CountVectorizer instance); you can pass it
    instead of ``feature_names``.

    ``coef_scale`` is a 1D np.ndarray with a scaling coefficient
    for each feature; coef[i] = coef[i] * coef_scale[i] if
    coef_scale[i] is not nan. Use it if you want to scale coefficients
    before displaying them, to take input feature sign or scale in account.
    """
    feature_names, coef_scale = handle_hashing_vec(vec, feature_names,
                                                   coef_scale)
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)
    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re)

    _extra_caveats = "\n" + HASHING_CAVEATS if is_invhashing(vec) else ''

    def _features(label_id):
        coef = get_coef(clf, label_id, scale=coef_scale)
        if flt_indices is not None:
            coef = coef[flt_indices]
        return get_top_features(feature_names, coef, top)

    display_names = get_target_display_names(clf.classes_, target_names, targets)
    if is_multiclass_classifier(clf):
        return Explanation(
            targets=[
                TargetExplanation(
                    target=label,
                    feature_weights=_features(label_id)
                )
                for label_id, label in display_names
                ],
            description=DESCRIPTION_CLF_MULTICLASS + _extra_caveats,
            estimator=repr(clf),
            method='linear model',
        )
    else:
        # for binary classifiers scikit-learn stores a single coefficient
        # vector, which corresponds to clf.classes_[1].
        return Explanation(
            targets=[
                TargetExplanation(
                    target=display_names[1][1],
                    feature_weights=_features(0),
                )
            ],
            description=DESCRIPTION_CLF_BINARY + _extra_caveats,
            estimator=repr(clf),
            method='linear model',
        )


@explain_weights_sklearn.register(RandomForestClassifier)
@explain_weights_sklearn.register(RandomForestRegressor)
@explain_weights_sklearn.register(ExtraTreesClassifier)
@explain_weights_sklearn.register(ExtraTreesRegressor)
@explain_weights_sklearn.register(GradientBoostingClassifier)
@explain_weights_sklearn.register(GradientBoostingRegressor)
@explain_weights_sklearn.register(AdaBoostClassifier)
@explain_weights_sklearn.register(AdaBoostRegressor)
def explain_rf_feature_importance(estimator,
                                  vec=None,
                                  top=_TOP,
                                  target_names=None,  # ignored
                                  targets=None,  # ignored
                                  feature_names=None,
                                  feature_re=None,
                                  feature_filter=None,
                                  ):
    """
    Return an explanation of a tree-based ensemble estimator.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``, ``feature_re`` and ``feature_filter``
    parameters.

    ``target_names`` and ``targets`` parameters are ignored.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the estimator (e.g. a fitted
    CountVectorizer instance); you can pass it instead of ``feature_names``.
    """
    feature_names = get_feature_names(estimator, vec,
                                      feature_names=feature_names)
    coef = estimator.feature_importances_
    trees = np.array(estimator.estimators_).ravel()
    coef_std = np.std([tree.feature_importances_ for tree in trees], axis=0)

    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re)
    if flt_indices is not None:
        coef = coef[flt_indices]
        coef_std = coef_std[flt_indices]

    indices = argsort_k_largest_positive(coef, top)
    names, values, std = feature_names[indices], coef[indices], coef_std[indices]
    return Explanation(
        feature_importances=FeatureImportances(
            [FeatureWeight(*x) for x in zip(names, values, std)],
            remaining=np.count_nonzero(coef) - len(indices),
        ),
        description=DESCRIPTION_RANDOM_FOREST,
        estimator=repr(estimator),
        method='feature importances',
    )


@explain_weights_sklearn.register(DecisionTreeClassifier)
@explain_weights_sklearn.register(DecisionTreeRegressor)
def explain_decision_tree(estimator,
                          vec=None,
                          top=_TOP,
                          target_names=None,
                          targets=None,  # ignored
                          feature_names=None,
                          feature_re=None,
                          feature_filter=None,
                          **export_graphviz_kwargs):
    """
    Return an explanation of a decision tree.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``target_names``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``targets`` parameter is ignored.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the estimator (e.g. a fitted
    CountVectorizer instance); you can pass it instead of ``feature_names``.

    All other keyword arguments are passed to
    `sklearn.tree.export_graphviz`_ function.

    .. _sklearn.tree.export_graphviz: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    """
    feature_names = get_feature_names(estimator, vec,
                                      feature_names=feature_names)
    coef = estimator.feature_importances_
    tree_feature_names = feature_names
    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re)
    if flt_indices is not None:
        coef = coef[flt_indices]
    indices = argsort_k_largest_positive(coef, top)
    names, values = feature_names[indices], coef[indices]
    export_graphviz_kwargs.setdefault("proportion", True)
    tree_info = get_tree_info(
        estimator,
        feature_names=tree_feature_names,
        class_names=target_names,
        **export_graphviz_kwargs)

    return Explanation(
        feature_importances=FeatureImportances(
            [FeatureWeight(*x) for x in zip(names, values)],
            remaining=np.count_nonzero(coef) - len(indices),
        ),
        decision_tree=tree_info,
        description=DESCRIPTION_DECISION_TREE,
        estimator=repr(estimator),
        method='decision tree',
    )


@explain_weights_sklearn.register(ElasticNet)
@explain_weights_sklearn.register(ElasticNetCV)
@explain_weights_sklearn.register(HuberRegressor)
@explain_weights_sklearn.register(Lars)
@explain_weights_sklearn.register(LassoCV)
@explain_weights_sklearn.register(LinearRegression)
@explain_weights_sklearn.register(LinearSVR)
@explain_weights_sklearn.register(OrthogonalMatchingPursuit)
@explain_weights_sklearn.register(OrthogonalMatchingPursuitCV)
@explain_weights_sklearn.register(PassiveAggressiveRegressor)
@explain_weights_sklearn.register(Ridge)
@explain_weights_sklearn.register(RidgeCV)
@explain_weights_sklearn.register(SGDRegressor)
@explain_weights_sklearn.register(TheilSenRegressor)
def explain_linear_regressor_weights(reg,
                                     vec=None,
                                     top=_TOP,
                                     target_names=None,
                                     targets=None,
                                     feature_names=None,
                                     coef_scale=None,
                                     feature_re=None,
                                     feature_filter=None,
                                     ):
    """
    Return an explanation of a linear regressor weights.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``target_names``, ``targets``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the regressor ``reg``; you can
    pass it instead of ``feature_names``.

    ``coef_scale`` is a 1D np.ndarray with a scaling coefficient
    for each feature; coef[i] = coef[i] * coef_scale[i] if
    coef_scale[i] is not nan. Use it if you want to scale coefficients
    before displaying them, to take input feature sign or scale in account.
    """
    feature_names, coef_scale = handle_hashing_vec(vec, feature_names,
                                                   coef_scale)
    feature_names = get_feature_names(reg, vec, feature_names=feature_names)
    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re)
    _extra_caveats = "\n" + HASHING_CAVEATS if is_invhashing(vec) else ''

    def _features(target_id):
        coef = get_coef(reg, target_id, scale=coef_scale)
        if flt_indices is not None:
            coef = coef[flt_indices]
        return get_top_features(feature_names, coef, top)

    display_names = get_target_display_names(get_default_target_names(reg),
                                             target_names, targets)
    if is_multitarget_regressor(reg):
        return Explanation(
            targets=[
                TargetExplanation(
                    target=target_name,
                    feature_weights=_features(target_id)
                )
                for target_id, target_name in display_names
                ],
            description=DESCRIPTION_REGRESSION_MULTITARGET + _extra_caveats,
            estimator=repr(reg),
            method='linear model',
            is_regression=True,
        )
    else:
        return Explanation(
            targets=[TargetExplanation(
                target=display_names[0][1],
                feature_weights=_features(0),
            )],
            description=DESCRIPTION_REGRESSION + _extra_caveats,
            estimator=repr(reg),
            method='linear model',
            is_regression=True,
        )
