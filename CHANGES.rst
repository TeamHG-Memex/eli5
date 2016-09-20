Changelog
=========

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
