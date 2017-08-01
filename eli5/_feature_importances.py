# -*- coding: utf-8 -*-
from eli5.base import FeatureImportances, Explanation
from eli5.sklearn.utils import get_feature_names_filtered
from eli5.utils import argsort_k_largest


def get_feature_importances_filtered(coef, feature_names, flt_indices, top,
                                     coef_std=None):
    # type: (...) -> FeatureImportances
    if flt_indices is not None:
        coef = coef[flt_indices]
        if coef_std is not None:
            coef_std = coef_std[flt_indices]

    indices = argsort_k_largest(coef, top)
    names, values = feature_names[indices], coef[indices]
    std = None if coef_std is None else coef_std[indices]
    return FeatureImportances.from_names_values(
        names, values, std,
        remaining=coef.shape[0] - len(indices),
    )


def get_feature_importance_explanation(estimator, vec, coef, feature_names,
                                       feature_filter, feature_re, top,
                                       description, is_regression,
                                       estimator_feature_names=None,
                                       num_features=None,
                                       coef_std=None):
    # type: (...) -> Explanation
    feature_names, flt_indices = get_feature_names_filtered(
        estimator, vec,
        feature_names=feature_names,
        estimator_feature_names=estimator_feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        num_features=num_features,
    )
    feature_importances = get_feature_importances_filtered(
        coef, feature_names, flt_indices, top, coef_std)
    return Explanation(
        feature_importances=feature_importances,
        description=description,
        estimator=repr(estimator),
        method='feature importances',
        is_regression=is_regression,
    )
