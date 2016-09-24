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
   :target: http://eli5.readthedocs.org/en/latest/?badge=latest
   :alt: Documentation


ELI5 is a Python package which helps to debug machine learning
classifiers and explain their predictions.

Currently it allows to:

* explain weights and predictions of scikit-learn linear classifiers
  and regressors;
* explain weights of scikit-learn decision trees and tree-based ensemble
  classifiers (via feature importances);
* debug scikit-learn pipelines which contain HashingVectorizer,
  by undoing hashing;
* explain predictions of any black-box classifier using LIME
  ( http://arxiv.org/abs/1602.04938 ) algorithm.

TODO:

* IPython and HTML support
* https://github.com/TeamHG-Memex/sklearn-crfsuite
  and https://github.com/tpeng/python-crfsuite
* https://github.com/scikit-learn-contrib/polylearn
* https://github.com/scikit-learn-contrib/lightning
* fasttext (?)
* xgboost (?)
* eli5.lime improvements
* image input
* built-in support for non-text data in eli5.lime
* tensorflow, theano, lasagne, keras
* Naive Bayes from scikit-learn
  (see https://github.com/scikit-learn/scikit-learn/issues/2237)
* Reinforcement Learning support
* explain predictions of decision trees and treee-based ensembles

License is MIT.

Check `docs <http://eli5.readthedocs.org/>`_ for more (sorry, also TODO).
