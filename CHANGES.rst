Changelog
=========

0.1 (TBA)
---------

* HTML output;
* IPython integration;
* JSON output;
* visualization of scikit-learn text vectorizers;
* `sklearn-crfsuite <https://github.com/TeamHG-Memex/sklearn-crfsuite>`_
  support;
* `lightning <https://github.com/scikit-learn-contrib/lightning>`_ support;
* top-level :func:`eli5.explain_weights` and :func:`eli5.explain_prediction`
  functions;
* :ref:`eli5.lime <eli5-lime>` improvements: samplers for non-text data,
  bug fixes, docs;
* HashingVectorizer is supported for regression tasks;
* performance improvements - feature names are lazy;
* sklearn ElasticNetCV and RidgeCV support;
* it is now possible to customize formatting output - show/hide sections,
  change layout;
* sklearn OneVsRestClassifier support;
* sklearn DecisionTreeClassifier visualization (text-based or svg-based);
* dropped support for scikit-learn < 0.18;
* basic mypy type annotations;
* ``feature_re`` argument of :func:`eli5.explain_weights`
  allows to show only a subset of features;
* ``target_names`` argument of :func:`eli5.explain_weights`
  allows to change display names of targets/classes;
* ``targets`` argument of :func:`eli5.explain_weights` allows to
  show a subset of targets/classes and change their display order;
* documentation.


0.0.6 (2016-10-12)
------------------

* Candidate features in eli5.sklearn.InvertableHashingVectorizer
  are ordered by their frequency, first candidate is always positive.

0.0.5 (2016-09-27)
------------------

* HashingVectorizer support in explain_prediction;
* add an option to pass coefficient scaling array; it is useful
  if you want to compare coefficients for features which scale or sign
  is different in the input;
* bug fix: classifier weights are no longer changed by eli5 functions.

0.0.4 (2016-09-24)
------------------

* eli5.sklearn.InvertableHashingVectorizer and
  eli5.sklearn.FeatureUnhasher allow to recover feature names for
  pipelines which use HashingVectorizer or FeatureHasher;
* added support for scikit-learn linear regression models (ElasticNet,
  Lars, Lasso, LinearRegression, LinearSVR, Ridge, SGDRegressor);
* doc and vec arguments are swapped in explain_prediction function;
  vec can now be omitted if an example is already vectorized;
* fixed issue with dense feature vectors;
* all class_names arguments are renamed to target_names;
* feature name guessing is fixed for scikit-learn ensemble estimators;
* testing improvements.

0.0.3 (2016-09-21)
------------------

* support any black-box classifier using LIME (http://arxiv.org/abs/1602.04938)
  algorithm; text data support is built-in;
* "vectorized" argument for sklearn.explain_prediction; it allows to pass
  example which is already vectorized;
* allow to pass feature_names explicitly;
* support classifiers without get_feature_names method using auto-generated
  feature names.

0.0.2 (2016-09-19)
------------------

* 'top' argument of ``explain_prediction``
  can be a tuple (num_positive, num_negative);
* classifier name is no longer printed by default;
* added eli5.sklearn.explain_prediction to explain individual examples;
* fixed numpy warning.

0.0.1 (2016-09-15)
------------------

Pre-release.
