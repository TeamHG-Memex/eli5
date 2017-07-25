.. _eli5-decrease-in-score:

Decrease in Score
=================

eli5 provides a way to compute feature importances for any black-box
estimator by measuring how score decreases when a feature is not available.

A similar method is described in Breiman, "Random Forests", Machine Learning,
45(1), 5-32, 2001 (available online at
https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf) as
"mean decrease in accuracy".

The idea is the following: feature importance can be measured by looking at
how much the score (accuracy, F1, R^2, etc. - any score we're interested in)
decreases when a feature is not available. To do that one can remove feature
from the dataset, re-train the estimator and check the score. But it requires
re-training an estimator for each feature, which can be computationally
intensive.

To avoid re-training the estimator we can remove a feature only from the
test part of the dataset, and compute score without using this
feature. It doesn't work as-is, because estimators expect feature to be
present. So instead of removing a feature we can replace it with random
noise - feature column is still there, but it no longer contains useful
information. This method works if noise is drawn from the same
distribution as original feature values (as otherwise estimator may
fail). The simplest way to get such noise is to shuffle values
for a feature, i.e. use other example's feature value - this is how
"decrease in score" feature importances computed.

For sklearn-compatible estimators eli5 provides
:class:`~.ScoreDecreaseFeatureImportances` wrapper; if you want to use this
method for other estimators there is :mod:`eli5.score_decrease` module.

For example, this is how you can check feature importances of
`sklearn.svm.SVC`_ classifier, which is not supported by eli5 directly
when a non-linear kernel is used::

    import eli5
    from eli5.sklearn import ScoreDecreaseFeatureImportances
    from sklearn.svm import SVC

    # ... load data

    svc = SVC().fit(X_train, y_train)
    sd = ScoreDecreaseFeatureImportances(svc).fit(X_test, y_test)
    eli5.show_weights(sd)

.. _sklearn.svm.SVC: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

If you don't have a separate held-out dataset, you can fit
:class:`~.ScoreDecreaseFeatureImportances` on the same data as used for
training; this still allows to inspect the model, but doesn't show which
features are important for generalization.

This method can be useful not only for introspection, but also for
model selection - one can compute feature importances using
:class:`~.ScoreDecreaseFeatureImportances`, then drop unimportant features
using e.g. sklearn's SelectFromModel_. In this case estimator passed to
:class:`~.ScoreDecreaseFeatureImportances` doesn't have to be fit; feature
importances can be computed for several train/test splits and then averaged::

    import eli5
    from eli5.sklearn import ScoreDecreaseFeatureImportances
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectFromModel

    # ... load data

    sd = ScoreDecreaseFeatureImportances(SVC(), cv=5)
    sd.fit(X, y)

    # sd.feature_importances_ attribute is now available, it can be used
    # for feature selection - let's e.g. select features which increase
    # accuracy by at least 0.05:
    sel = SelectFromModel(sd, threshold=0.05, prefit=True)
    X_trans = sel.transform(X)

    # It is possible to combine SelectFromModel and
    # ScoreDecreaseFeatureImportances directly, without fitting
    # ScoreDecreaseFeatureImportances first:
    sel = SelectFromModel(
        ScoreDecreaseFeatureImportances(SVC(), cv=5),
        threshold=0.05,
    ).fit(X, y)
    X_trans = sel.transform(X)

See :class:`~.ScoreDecreaseFeatureImportances` docs for more.

.. _SelectFromModel: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel