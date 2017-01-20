Changelog
=========

0.4 (2017-01-20)
----------------

* :func:`eli5.explain_prediction`: new 'top_targets' argument allows
  to display only predictions with highest or lowest scores;
* :func:`eli5.explain_weights` allows to customize the way feature importances
  are computed for XGBClassifier and XGBRegressor using ``importance_type``
  argument (see docs for the :ref:`eli5 XGBoost support <library-xgboost>`);
* :func:`eli5.explain_weights` uses gain for XGBClassifier and XGBRegressor
  feature importances by default; this method is a better indication of
  what's going, and it makes results more compatible with feature importances
  displayed for scikit-learn gradient boosting methods.

0.3.1 (2017-01-16)
------------------

* packaging fix: scikit-learn is added to install_requires in setup.py.

0.3 (2017-01-13)
----------------

* :func:`eli5.explain_prediction` works for XGBClassifier, XGBRegressor
  from XGBoost and for ExtraTreesClassifier, ExtraTreesRegressor,
  GradientBoostingClassifier, GradientBoostingRegressor,
  RandomForestClassifier, RandomForestRegressor, DecisionTreeClassifier
  and DecisionTreeRegressor from scikit-learn.
  Explanation method is based on
  http://blog.datadive.net/interpreting-random-forests/ .
* :func:`eli5.explain_weights` now supports tree-based regressors from
  scikit-learn: DecisionTreeRegressor, AdaBoostRegressor,
  GradientBoostingRegressor, RandomForestRegressor and ExtraTreesRegressor.
* :func:`eli5.explain_weights` works for XGBRegressor;
* new :ref:`TextExplainer <lime-tutorial>` class allows to explain predictions
  of black-box text classification pipelines using LIME algorithm;
  many improvements in :ref:`eli5.lime <eli5-lime>`.
* better ``sklearn.pipeline.FeatureUnion`` support in
  :func:`eli5.explain_prediction`;
* rendering performance is improved;
* a number of remaining feature importances is shown when the feature
  importance table is truncated;
* styling of feature importances tables is fixed;
* :func:`eli5.explain_weights` and :func:`eli5.explain_prediction` support
  more linear estimators from scikit-learn: HuberRegressor, LarsCV, LassoCV,
  LassoLars, LassoLarsCV, LassoLarsIC, OrthogonalMatchingPursuit,
  OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor,
  RidgeClassifier, RidgeClassifierCV, TheilSenRegressor.
* text-based formatting of decision trees is changed: for binary
  classification trees only a probability of "true" class is printed,
  not both probabilities as it was before.
* :func:`eli5.explain_weights` supports ``feature_filter`` in addition
  to ``feature_re`` for filtering features, and :func:`eli5.explain_prediction`
  now also supports both of these arguments;
* 'Weight' column is renamed to 'Contribution' in the output of
  :func:`eli5.explain_prediction`;
* new ``show_feature_values=True`` formatter argument allows to display
  input feature values;
* fixed an issue with analyzer='char_wb' highlighting at the start of the
  text.

0.2 (2016-12-03)
----------------

* XGBClassifier support (from `XGBoost <https://github.com/dmlc/xgboost>`__
  package);
* :func:`eli5.explain_weights` support for sklearn OneVsRestClassifier;
* std deviation of feature importances is no longer printed as zero
  if it is not available.

0.1.1 (2016-11-25)
------------------

* packaging fixes: require attrs > 16.0.0, fixed README rendering

0.1 (2016-11-24)
----------------

* HTML output;
* IPython integration;
* JSON output;
* visualization of scikit-learn text vectorizers;
* `sklearn-crfsuite <https://github.com/TeamHG-Memex/sklearn-crfsuite>`__
  support;
* `lightning <https://github.com/scikit-learn-contrib/lightning>`__ support;
* :func:`eli5.show_weights` and :func:`eli5.show_prediction` functions;
* :func:`eli5.explain_weights` and :func:`eli5.explain_prediction`
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
* ``feature_re`` argument allows to show only a subset of features;
* ``target_names`` argument allows to change display names of targets/classes;
* ``targets`` argument allows to show a subset of targets/classes and
  change their display order;
* documentation, more examples.


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
