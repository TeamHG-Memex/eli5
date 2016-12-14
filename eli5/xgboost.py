# -*- coding: utf-8 -*-
from __future__ import absolute_import
from singledispatch import singledispatch

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from eli5.base import FeatureWeight, FeatureImportances, Explanation
from eli5.explain import explain_weights
from eli5.sklearn.utils import get_feature_names
from eli5.utils import argsort_k_largest


DESCRIPTION_XGBOOST = """
XGBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


@explain_weights.register(XGBClassifier)
@explain_weights.register(XGBRegressor)
@singledispatch
def explain_weights_xgboost(xgb,
                            vec=None,
                            top=20,
                            target_names=None,  # ignored
                            targets=None,  # ignored
                            feature_names=None,
                            feature_re=None):
    """
    Return an explanation of an XGBoost estimator (via scikit-learn wrapper).
    """
    feature_names = get_feature_names(xgb, vec, feature_names=feature_names)
    coef = xgb.feature_importances_

    if feature_re is not None:
        feature_names, flt_indices = feature_names.filtered_by_re(feature_re)
        coef = coef[flt_indices]

    indices = argsort_k_largest(coef, top)
    names, values = feature_names[indices], coef[indices]
    return Explanation(
        feature_importances=FeatureImportances(
            [FeatureWeight(*x) for x in zip(names, values)],
            remaining=np.count_nonzero(coef) - len(indices),
        ),
        description=DESCRIPTION_XGBOOST,
        estimator=repr(xgb),
        method='feature importances',
    )
