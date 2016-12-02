====
ELI5
====

.. image:: https://img.shields.io/pypi/v/eli5.svg
   :target: https://pypi.python.org/pypi/eli5
   :alt: PyPI Version

.. image:: https://travis-ci.org/TeamHG-Memex/eli5.svg?branch=master
   :target: http://travis-ci.org/TeamHG-Memex/eli5
   :alt: Build Status

.. image:: http://codecov.io/github/TeamHG-Memex/eli5/coverage.svg?branch=master
   :target: http://codecov.io/github/TeamHG-Memex/eli5?branch=master
   :alt: Code Coverage

.. image:: https://readthedocs.org/projects/eli5/badge/?version=latest
   :target: http://eli5.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation


ELI5 is a Python package which helps to debug machine learning
classifiers and explain their predictions.

.. image:: https://raw.githubusercontent.com/TeamHG-Memex/eli5/master/docs/source/static/word-highlight.png
   :alt: explain_prediction for text data

It provides support for the following machine learning frameworks and packages:

* scikit-learn_. Currently ELI5 allows to explain weights and predictions
  of scikit-learn linear classifiers and regressors, print decision trees
  as text or as SVG, show feature importances of random forests. ELI5
  understands text processing utilities from scikit-learn and can highlight
  text data accordingly. It also allows to debug scikit-learn pipelines which
  contain HashingVectorizer, by undoing hashing.

* lightning_ - explain weights and predictions of lightning classifiers and
  regressors.

* sklearn-crfsuite_. ELI5 allows to check weights of sklearn_crfsuite.CRF
  models.

* xgboost_ - show feature importances using the same interface.

ELI5 also provides an alternative implementation of LIME_ algorithm,
which allows to explain predictions of any black-box classifier. This feature
is currently experimental.

Explanation and formatting are separated; you can get text-based explanation
to display in console, HTML version embeddable in an IPython notebook
or web dashboards, or JSON version which allows to implement custom
rendering and formatting on a client.

.. _lightning: https://github.com/scikit-learn-contrib/lightning
.. _scikit-learn: https://github.com/scikit-learn/scikit-learn
.. _sklearn-crfsuite: https://github.com/TeamHG-Memex/sklearn-crfsuite
.. _LIME: http://eli5.readthedocs.io/en/latest/lime.html
.. _xgboost: https://github.com/dmlc/xgboost

License is MIT.

Check `docs <http://eli5.readthedocs.io/>`_ for more.
