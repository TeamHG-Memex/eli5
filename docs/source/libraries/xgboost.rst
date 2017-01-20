.. _library-xgboost:

XGBoost
=======

XGBoost_ is a popular Gradient Boosting library with Python interface.
eli5 supports :func:`eli5.explain_weights` and :func:`eli5.explain_prediction`
for XGBClassifer_ and XGBRegressor_ estimators. It is tested for
xgboost >= 0.6a2.

.. _XGBoost: https://github.com/dmlc/xgboost
.. _XGBClassifer: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
.. _XGBRegressor: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor

:func:`eli5.explain_weights` uses feature importances. Additional
arguments for XGBClassifer_ and XGBRegressor_:

* ``importance_type`` is a way to get feature importance. Possible values are:

  - 'gain' - the average gain of the feature when it is used in trees
    (default)
  - 'weight' - the number of times a feature is used to split the data
    across all trees
  - 'cover' - the average coverage of the feature when it is used in trees

``target_names`` and ``target`` arguments are ignored.

.. note::
    Top-level :func:`eli5.explain_weights` calls are dispatched
    to :func:`eli5.xgboost.explain_weights_xgboost` for
    XGBClassifer_ and XGBRegressor_.

For :func:`eli5.explain_prediction` eli5 uses an approach based on ideas from
http://blog.datadive.net/interpreting-random-forests/ :
feature weights are calculated by following decision paths in trees
of an ensemble. Each node of the tree has an output score, and
contribution of a feature on the decision path is how much the score changes
from parent to child.

Additional :func:`eli5.explain_prediction` keyword arguments supported
for XGBClassifer_ and XGBRegressor_:

* ``vec`` is a vectorizer instance used to transform
  raw features to the input of the estimator ``xgb``
  (e.g. a fitted CountVectorizer instance); you can pass it
  instead of ``feature_names``.

* ``vectorized`` is a flag which tells eli5 if ``doc`` should be
  passed through ``vec`` or not. By default it is False, meaning that
  if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
  estimator. Set it to False if you're passing ``vec``,
  but ``doc`` is already vectorized.

See the :ref:`tutorial <xgboost-titanic-tutorial>` for a more detailed usage
example.

.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.xgboost.explain_prediction_xgboost` for
    XGBClassifer_ and XGBRegressor_.

