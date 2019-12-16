from __future__ import absolute_import, division

import numpy as np
import catboost

from eli5.explain import explain_weights
from eli5._feature_importances import get_feature_importance_explanation

DESCRIPTION_CATBOOST = """CatBoost feature importances; 
values are numbers 0 <= x <= 1; all values sum to 1."""

@explain_weights.register(catboost.CatBoost)
@explain_weights.register(catboost.CatBoostClassifier)
@explain_weights.register(catboost.CatBoostRegressor)
def explain_weights_catboost(catb, 
                             vec=None,
                             top=20,
                             importance_type='PredictionValuesChange',
                             feature_names=None,
                             pool=None
                             ):
    """
    Return an explanation of an CatBoost estimator (CatBoostClassifier,
    CatBoost, CatBoostRegressor) as feature importances.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``target_names`` and ``targets`` parameters are ignored.

    Parameters
    ----------
    :param 'importance_type' : str, optional
        A way to get feature importance. Possible values are:
        
        - 'PredictionValuesChange' (default) - The individual importance
          values for each of the input features.
        - 'LossFunctionChange' - The individual importance values for
          each of the input features for ranking metrics
          (requires training data to be passed or a similar dataset with Pool)

    :param 'pool' : catboost.Pool, optional
        To be passed if explain_weights_catboost has importance_type set
        to LossFunctionChange. The catboost feature_importances uses the Pool
        datatype to calculate the parameter for the specific importance_type.
    """
    is_regression = _is_regression(catb)
    catb_feature_names = catb.feature_names_
    coef = _catb_feature_importance(catb, importance_type=importance_type, pool=pool)
    return get_feature_importance_explanation(catb, vec, coef,
                                              feature_names=feature_names,
                                              estimator_feature_names=catb_feature_names,
                                              feature_filter=None,
                                              feature_re=None,
                                              top=top,
                                              description=DESCRIPTION_CATBOOST,
                                              is_regression=is_regression,
                                              num_features=coef.shape[-1]
                                              )


def _is_regression(catb):
    return isinstance(catb, catboost.CatBoostRegressor)


def _catb_feature_importance(catb, importance_type, pool=None):
    if importance_type == "PredictionValuesChange":
        fs = catb.get_feature_importance(type=importance_type)
    elif importance_type == "LossFunctionChange":
        if isinstance(pool, catboost.Pool):
            fs = catb.get_feature_importance(data=pool, type=importance_type)
        else:
            raise ValueError(
                'importance_type: "LossFunctionChange" requires catboost.Pool '
                'datatype to be passed with parameter pool to calculate '
                'metric. Either no datatype or invalid datatype was passed'
            )
    else:
        raise ValueError(
            'Only two importance_type "PredictionValuesChange" '
            'and "LossFunctionChange" are supported. Invalid Parameter '
            '{} for importance_type'.format(importance_type))
    all_features = np.array(fs, dtype=np.float32)
    return all_features/all_features.sum()