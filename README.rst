====
ELI5
====

.. image:: https://img.shields.io/pypi/v/eli5.svg
   :target: https://pypi.python.org/pypi/eli5
   :alt: PyPI Version

.. image:: https://travis-ci.org/TeamHG-Memex/eli5.svg?branch=master
   :target: https://travis-ci.org/TeamHG-Memex/eli5
   :alt: Build Status

.. image:: https://codecov.io/github/TeamHG-Memex/eli5/coverage.svg?branch=master
   :target: https://codecov.io/github/TeamHG-Memex/eli5?branch=master
   :alt: Code Coverage

.. image:: https://readthedocs.org/projects/eli5/badge/?version=latest
   :target: https://eli5.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation


ELI5 is a Python package which helps to debug machine learning
classifiers and explain their predictions.

.. image:: https://raw.githubusercontent.com/TeamHG-Memex/eli5/master/docs/source/static/word-highlight.png
   :alt: explain_prediction for text data

It provides support for the following machine learning frameworks and packages:

* scikit-learn_. Currently ELI5 allows to explain weights and predictions
  of scikit-learn linear classifiers and regressors, print decision trees
  as text or as SVG, show feature importances and explain predictions
  of decision trees and tree-based ensembles. ELI5 understands text
  processing utilities from scikit-learn and can highlight text data
  accordingly. Pipeline and FeatureUnion are supported.
  It also allows to debug scikit-learn pipelines which contain
  HashingVectorizer, by undoing hashing.

* xgboost_ - show feature importances and explain predictions of XGBClassifier,
  XGBRegressor and xgboost.Booster.

* LightGBM_ - show feature importances and explain predictions of
  LGBMClassifier and LGBMRegressor.

* CatBoost_ - show feature importances of CatBoostClassifier,
  CatBoostRegressor and catboost.CatBoost.

* lightning_ - explain weights and predictions of lightning classifiers and
  regressors.

* sklearn-crfsuite_. ELI5 allows to check weights of sklearn_crfsuite.CRF
  models.

ELI5 also implements several algorithms for inspecting black-box models
(see `Inspecting Black-Box Estimators`_):

* TextExplainer_ allows to explain predictions
  of any text classifier using LIME_ algorithm (Ribeiro et al., 2016).
  There are utilities for using LIME with non-text data and arbitrary black-box
  classifiers as well, but this feature is currently experimental.
* `Permutation importance`_ method can be used to compute feature importances
  for black box estimators.

Explanation and formatting are separated; you can get text-based explanation
to display in console, HTML version embeddable in an IPython notebook
or web dashboards, a ``pandas.DataFrame`` object if you want to process
results further, or JSON version which allows to implement custom rendering
and formatting on a client.

.. _lightning: https://github.com/scikit-learn-contrib/lightning
.. _scikit-learn: https://github.com/scikit-learn/scikit-learn
.. _sklearn-crfsuite: https://github.com/TeamHG-Memex/sklearn-crfsuite
.. _LIME: https://eli5.readthedocs.io/en/latest/blackbox/lime.html
.. _TextExplainer: https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html
.. _xgboost: https://github.com/dmlc/xgboost
.. _LightGBM: https://github.com/Microsoft/LightGBM
.. _Catboost: https://github.com/catboost/catboost
.. _Permutation importance: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
.. _Inspecting Black-Box Estimators: https://eli5.readthedocs.io/en/latest/blackbox/index.html

License is MIT.

Check `docs <https://eli5.readthedocs.io/>`_ for more.

----

.. image:: https://hyperiongray.s3.amazonaws.com/define-hg.svg
	:target: https://www.hyperiongray.com/?pk_campaign=github&pk_kwd=eli5
	:alt: define hyperiongray
