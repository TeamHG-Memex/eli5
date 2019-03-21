.. _library-lightgbm:

LightGBM
========

LightGBM_ is a fast Gradient Boosting framework; it provides a Python
interface. eli5 supports :func:`eli5.explain_weights`
and :func:`eli5.explain_prediction` for ``lightgbm.LGBMClassifer``
and ``lightgbm.LGBMRegressor`` estimators.

.. _LightGBM: https://github.com/Microsoft/LightGBM

:func:`eli5.explain_weights` uses feature importances. Additional
arguments for LGBMClassifier and LGBMClassifier:

* ``importance_type`` is a way to get feature importance. Possible values are:

  - 'gain' - the average gain of the feature when it is used in trees
    (default)
  - 'split' - the number of times a feature is used to split the data
    across all trees
  - 'weight' - the same as 'split', for better compatibility with
    :ref:`library-xgboost`.

``target_names`` and ``target`` arguments are ignored.

.. note::
    Top-level :func:`eli5.explain_weights` calls are dispatched
    to :func:`eli5.lightgbm.explain_weights_lightgbm` for
    ``lightgbm.LGBMClassifer`` and ``lightgbm.LGBMRegressor``.

For :func:`eli5.explain_prediction` eli5 uses an approach based on ideas from
http://blog.datadive.net/interpreting-random-forests/ :
feature weights are calculated by following decision paths in trees
of an ensemble. Each node of the tree has an output score, and
contribution of a feature on the decision path is how much the score changes
from parent to child.

Additional :func:`eli5.explain_prediction` keyword arguments supported
for ``lightgbm.LGBMClassifer`` and ``lightgbm.LGBMRegressor``:

* ``vec`` is a vectorizer instance used to transform
  raw features to the input of the estimator ``lgb``
  (e.g. a fitted CountVectorizer instance); you can pass it
  instead of ``feature_names``.

* ``vectorized`` is a flag which tells eli5 if ``doc`` should be
  passed through ``vec`` or not. By default it is False, meaning that
  if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
  estimator. Set it to True if you're passing ``vec``,
  but ``doc`` is already vectorized.

.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.xgboost.explain_prediction_lightgbm` for
    ``lightgbm.LGBMClassifer`` and ``lightgbm.LGBMRegressor``.

