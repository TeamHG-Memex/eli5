Overview
========

ELI5 is a Python package which helps to debug machine learning
classifiers and explain their predictions. It provides support for the
following machine learning frameworks and packages:

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
.. _LIME: http://arxiv.org/abs/1602.04938

License is MIT.
