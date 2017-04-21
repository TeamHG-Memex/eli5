# -*- coding: utf-8 -*-
from __future__ import absolute_import
from singledispatch import singledispatch

import lightgbm  # type: ignore
from eli5.explain import explain_weights
from eli5._feature_importances import get_feature_importance_explanation


DESCRIPTION_LIGHTGBM = """
LightGBM feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


@explain_weights.register(lightgbm.LGBMClassifier)
@explain_weights.register(lightgbm.LGBMRegressor)
@singledispatch
def explain_weights_lightgbm(lgb,
                             vec=None,
                             top=20,
                             target_names=None,  # ignored
                             targets=None,  # ignored
                             feature_names=None,
                             feature_re=None,
                             feature_filter=None,
                             ):
    """
    Return an explanation of an LightGBM estimator (via scikit-learn wrapper
    LGBMClassifier or LGBMRegressor) as feature importances.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``target_names`` and ``targets`` parameters are ignored.
    """
    coef = lgb.feature_importances_
    return get_feature_importance_explanation(lgb, vec, coef,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        description=DESCRIPTION_LIGHTGBM,
        num_features=coef.shape[-1],
        is_regression=isinstance(lgb, lightgbm.LGBMRegressor),
    )
