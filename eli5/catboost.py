import numpy as np
from catboost import (
    CatBoost,
    CatBoostClassifier,
    CatBoostRegressor,
    Pool
)

from eli5.explain import explain_weights
from eli5._feature_importances import get_feature_importance_explanation

DESCRIPTION_CATBOOST = """
CatBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


@explain_weights.register(CatBoost)
@explain_weights.register(CatBoostClassifier)
@explain_weights.register(CatBoostRegressor)
def explain_weights_catboost(catb,
                            vec=None,
                            top=20,
                            importance_type='PredictionValuesChange',
                            prettified=True,
                            feature_names=None,
                            Pool=None
                            ):
    """
    Return an explanation of an CatBoost estimator (CatBoostClassifier, CatBoost, CatBoostRegressor)
    as feature importances.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``target_names`` and ``targets`` parameters are ignored.

    Parameters
    ----------
    importance_type : str, optional
        A way to get feature importance. Possible values are:

        - 'PredictionValuesChange' - The individual importance values for each of the input features.
          (default)
        - 'LossFunctionChange' - The individual importance values for each of the input features for ranking metrics (requires training data to be passed  or a similar dataset with Pool)
    prettified : bool, optional
        A way to get the feature names from the columns and display it against the feature importance
        - 'True' - return list of tuples on get_feature_importance on catboost objects along with the column names. (default)
        - 'False' - return list numbers on get_feature_importance on catboost objects.
    
    """ 
    is_regression = _check_catboost_args(catb)
    catb_feature_names = catb.feature_names_
    coef = _catb_feature_importance(catb,importance_type=importance_type,Pool=Pool)
    return get_feature_importance_explanation(catb,vec,coef,
                                                feature_names=feature_names,
                                                estimator_feature_names=catb_feature_names,
                                                feature_filter=None,
                                                feature_re=None,
                                                top=top,
                                                description=DESCRIPTION_CATBOOST,
                                                is_regression=is_regression,
                                                num_features=coef.shape[-1]
                                                )



def _check_catboost_args(catb):
    return isinstance(catb,CatBoostRegressor)


def _catb_feature_importance(catb,importance_type,Pool=None):
    if(importance_type=="PredictionValuesChange"):
        fs = catb.get_feature_importance(type=importance_type)
    elif(importance_type=="LossFunctionChange"):
        if(isinstance(pool,Pool)):
            fs = catb.get_feature_importance(Pool,type=importance_type)
        else:
            raise ValueError(
                'POOL datatype required or type:LossFunctionChange '
            )
    else:
        raise ValueError('invalid importance_type')
    all_features = np.array([*fs],dtype=np.float32)
    return all_features/all_features.sum()
