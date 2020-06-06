# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    SVC,
    SVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
# TODO: see https://github.com/scikit-learn/scikit-learn/pull/2250
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from eli5.base import (
    Explanation, TargetExplanation, FeatureImportances)
from eli5.base_utils import singledispatch
from eli5._feature_weights import get_top_features
from eli5.utils import argsort_k_largest_positive, get_target_display_names
from eli5.sklearn.unhashing import handle_hashing_vec, is_invhashing
from eli5.sklearn.treeinspect import get_tree_info
from eli5.sklearn.utils import (
    get_coef,
    is_multiclass_classifier,
    is_multitarget_regressor,
    get_feature_names,
    get_feature_names_filtered,
    get_default_target_names,
)
from eli5.explain import explain_weights
from eli5.transform import transform_feature_names
from eli5._feature_importances import (
    get_feature_importances_filtered,
    get_feature_importance_explanation,
)
from .permutation_importance import PermutationImportance


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

DESCRIPTION_SCORE_DECREASE = """
Feature importances, computed as a decrease in score when feature
values are permuted (i.e. become noise). This is also known as 
permutation importance.
"""

_TOP = 20


@singledispatch
def explain_weights_sklearn(estimator, vec=None, top=_TOP,
                            target_names=None,
                            targets=None,
                            feature_names=None, coef_scale=None,
                            feature_re=None, feature_filter=None):
    """ Return an explanation of an estimator """
    return explain_weights_sklearn_not_supported(estimator)


@explain_weights.register(BaseEstimator)
def explain_weights_sklearn_not_supported(
        estimator, vec=None, top=_TOP,
        target_names=None,
        targets=None,
        feature_names=None, coef_scale=None,
        feature_re=None, feature_filter=None):
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


def register(cls):
    def deco(f):
        return explain_weights.register(cls)(
            explain_weights_sklearn.register(cls)(f))
    return deco


@explain_weights.register(OneVsRestClassifier)
def explain_weights_ovr(ovr, **kwargs):
    estimator = ovr.estimator
    func = explain_weights.dispatch(estimator.__class__)
    return func(ovr, **kwargs)


@explain_weights_sklearn.register(OneVsRestClassifier)
def explain_weights_ovr_sklearn(ovr, **kwargs):
    # dispatch OvR to eli5.sklearn
    # if explain_weights_sklearn is called explicitly
    estimator = ovr.estimator
    func = explain_weights_sklearn.dispatch(estimator.__class__)
    return func(ovr, **kwargs)


@register(LogisticRegression)
@register(LogisticRegressionCV)
@register(SGDClassifier)
@register(PassiveAggressiveClassifier)
@register(Perceptron)
@register(LinearSVC)
@register(RidgeClassifier)
@register(RidgeClassifierCV)
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
    feature_names, flt_indices = get_feature_names_filtered(
        clf, vec,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
    )

    _extra_caveats = "\n" + HASHING_CAVEATS if is_invhashing(vec) else ''

    def _features(label_id):
        coef = get_coef(clf, label_id, scale=coef_scale)
        if flt_indices is not None:
            coef = coef[flt_indices]
        return get_top_features(feature_names, coef, top)

    classes = getattr(clf, "classes_", ["-1", "1"])  # OneClassSVM support
    display_names = get_target_display_names(classes, target_names, targets)
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


@register(SVC)
@register(NuSVC)
@register(OneClassSVM)
def explain_libsvm_linear_classifier_weights(clf, *args, **kwargs):
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
    return explain_linear_classifier_weights(clf, *args, **kwargs)



@register(RandomForestClassifier)
@register(RandomForestRegressor)
@register(ExtraTreesClassifier)
@register(ExtraTreesRegressor)
@register(GradientBoostingClassifier)
@register(GradientBoostingRegressor)
@register(AdaBoostClassifier)
@register(AdaBoostRegressor)
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
    coef = estimator.feature_importances_
    trees = np.array(estimator.estimators_).ravel()
    coef_std = np.std([tree.feature_importances_ for tree in trees], axis=0)
    return get_feature_importance_explanation(estimator, vec, coef,
        coef_std=coef_std,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        description=DESCRIPTION_RANDOM_FOREST,
        is_regression=isinstance(estimator, RegressorMixin),
    )


@register(DecisionTreeClassifier)
@register(DecisionTreeRegressor)
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
    tree_feature_names = feature_names
    feature_names, flt_indices = feature_names.handle_filter(
        feature_filter, feature_re)
    feature_importances = get_feature_importances_filtered(
        estimator.feature_importances_, feature_names, flt_indices, top)

    export_graphviz_kwargs.setdefault("proportion", True)
    tree_info = get_tree_info(
        estimator,
        feature_names=tree_feature_names,
        class_names=target_names,
        **export_graphviz_kwargs)

    return Explanation(
        feature_importances=feature_importances,
        decision_tree=tree_info,
        description=DESCRIPTION_DECISION_TREE,
        estimator=repr(estimator),
        method='decision tree',
    )


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
    if isinstance(reg, (SVR, NuSVR)) and reg.kernel != 'linear':
        return explain_weights_sklearn_not_supported(reg)

    feature_names, coef_scale = handle_hashing_vec(vec, feature_names,
                                                   coef_scale)
    feature_names, flt_indices = get_feature_names_filtered(
        reg, vec,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
    )
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


@register(Pipeline)
def explain_weights_pipeline(estimator, feature_names=None, **kwargs):
    last_estimator = estimator.steps[-1][1]
    transform_pipeline = Pipeline(estimator.steps[:-1])
    if 'vec' in kwargs:
        feature_names = get_feature_names(feature_names, vec=kwargs.pop('vec'))
    feature_names = transform_feature_names(transform_pipeline, feature_names)
    out = explain_weights(last_estimator,
                          feature_names=feature_names,
                          **kwargs)
    out.estimator = repr(estimator)
    return out


@register(PermutationImportance)
def explain_permutation_importance(estimator,
                                   vec=None,
                                   top=_TOP,
                                   target_names=None,  # ignored
                                   targets=None,  # ignored
                                   feature_names=None,
                                   feature_re=None,
                                   feature_filter=None,
                                   ):
    """
    Return an explanation of PermutationImportance.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``, ``feature_re`` and ``feature_filter``
    parameters.

    ``target_names`` and ``targets`` parameters are ignored.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the estimator (e.g. a fitted
    CountVectorizer instance); you can pass it instead of ``feature_names``.
    """
    coef = estimator.feature_importances_
    coef_std = estimator.feature_importances_std_
    return get_feature_importance_explanation(estimator, vec, coef,
        coef_std=coef_std,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        description=DESCRIPTION_SCORE_DECREASE + estimator.caveats_,
        is_regression=isinstance(estimator.wrapped_estimator_, RegressorMixin),
    )
