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

It can explain weights and predictions of:

* scikit-learn linear classifiers;
* scikit-learn decision trees and tree-based ensemble classifiers;
* any black-box classifier using LIME ( http://arxiv.org/abs/1602.04938 )
  algorithm.

TODO:

* https://github.com/TeamHG-Memex/sklearn-crfsuite
  and https://github.com/tpeng/python-crfsuite
* https://github.com/scikit-learn-contrib/polylearn
* https://github.com/scikit-learn-contrib/lightning
* fasttext (?)
* xgboost (?)
* image input
* built-in support for non-text data in eli5.lime;
* tensorflow, theano, lasagne, keras
* Naive Bayes from scikit-learn
  (see https://github.com/scikit-learn/scikit-learn/issues/2237)
* eli5.lime improvements;
* IPython and HTML support;
* regression models;
* Reinforcement Learning support.

License is MIT.

Check `docs <http://eli5.readthedocs.org/>`_ for more (sorry, also TODO).
