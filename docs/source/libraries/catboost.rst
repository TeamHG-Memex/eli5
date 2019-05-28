.. _library-catboost:

CatBoost 
========

CatBoost_ is a state-of-the-art open-source gradient boosting on decision trees library. eli5 supports :func:`eli5.explain_weights`
for ``catboost.CatBoost``, ``catboost.CatBoostClassifier`` and ``catboost.CatBoostRegressor``.

.. _CatBoost: https://github.com/catboost/catboost

:func:`eli5.explain_weights` uses feature importances. Additional
arguments for CatBoostClassifier and CatBoostRegressor:

* ``importance_type`` is a way to get feature importance. Possible values are:
  
  - 'PredictionValuesChange' - The individual importance values for each of the input features.(default)
  - 'LossFunctionChange' - The individual importance values for each of the input features for ranking metrics (requires training data to be passed  or a similar dataset with Pool)

* ``pool`` the ``catboost.Pool`` datatype . To be passed if ``explain_weights_catboost`` has importance_type set to 'LossFunctionChange'. The catboost ``feature_importances`` uses the ``Pool`` datatype to calculate the parameter for the specific ``importance_type``.

.. note::
    Top-level :func:`eli5.explain_weights` calls are dispatched
    to :func:`eli5.catboost.explain_weights_catboost` for
    ``catboost.CatBoost``, ``catboost.CatBoostClassifer`` and ``catboost.CatBoostRegressor``.

