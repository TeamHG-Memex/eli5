"""
A sklearn-compatible object for computing score decrease-based
feature importances.
"""
from functools import partial
from typing import List

import numpy as np  # type: ignore

from sklearn.model_selection import check_cv  # type: ignore
from sklearn.utils.metaestimators import if_delegate_has_method  # type: ignore
from sklearn.utils import check_array, check_random_state  # type: ignore
from sklearn.base import (  # type: ignore
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier
)
from sklearn.metrics.scorer import check_scoring  # type: ignore

from eli5 import score_decrease


class ScoreDecreaseFeatureImportances(BaseEstimator, MetaEstimatorMixin):
    """Meta-transformer which exposes feature_importances_ attribute
    based on average score decrease.

    ScoreDecreaseFeatureImportances instance can be used instead of
    its wrapped estimator, as it exposes all estimator's common methods like
    ``predict``.

    There are 3 main modes of operation:

    1. prefit=True (pre-fit estimator is passed). You can call
       ScoreDecreaseFeatureImportances.fit either with training data, or
       with a held-out dataset (in the latter case ``feature_importances_``
       would be importances of features for generalization). After the fitting
       ``feature_importances_`` attribute becomes available, but the estimator
       itself is not fit again.
    2. prefit=False, cv=None. In this case ``fit`` method fits
       the estimator and computes feature importances on the same data,
       i.e. feature importances don't reflect importance of features
       for generalization.
    3. prefit=False, cv is not None. ``fit`` method fits the estimator, but
       instead of computing feature importances for the concrete estimator
       which is fit, importances are computed for a sequence of estimators
       trained and evaluated on train/test splits according to ``cv``, and
       then averaged. This is more resource-intensive (estimators are fit
       multiple times), and importances are not computed for the final
       estimator, but ``feature_importances_`` show importances of features
       for generalization.

    Mode (1) is most useful for inspecting an existing estimator; modes
    (2) and (3) can be also used for feature selection, together with
    sklearn's SelectFromModel.

    Parameters
    ----------
    estimator : object
        The base estimator. This can be both a fitted
        (if ``prefit`` is set to True) or a non-fitted estimator.

    prefit : bool, default True
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``fit`` must be called directly
        and ScoreDecreaseFeatureImportances cannot be used with
        ``cross_val_score``, ``GridSearchCV`` and similar utilities that clone
        the estimator.

    scoring : string, callable or None, default=None
        Scoring function to use for computing feature importances.
        A string with scoring name (see scikit-learn docs) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    n_iter : int, default 5
        Number of random shuffle iterations. Decrease to improve speed,
        increase to get more precise estimates.

    random_state : integer or numpy.random.RandomState, optional
        random state

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to disable cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        ``cv`` must be None if ``prefit`` is True.

    Attributes
    ----------
    feature_importances_ : array
        Feature importances, computed as mean decrease of the score.

    feature_importances_std_ : array
        Standard deviations of feature importances.

    results_ : list of arrays
        A list of score decreases for all experiments.

    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``ScoreDecreaseFeatureImportances``, i.e when prefit is False.

    rng_ : numpy.random.RandomState
        random state
    """
    def __init__(self, estimator, prefit=True, scoring=None, n_iter=5,
                 random_state=None, cv=None):
        if prefit and cv is not None:
            raise ValueError("cv must be None when prefit is True")
        self.estimator = estimator
        self.prefit = prefit
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.rng_ = check_random_state(random_state)

    def fit(self, X, y, groups=None, **fit_params):
        """Compute ``feature_importances_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if not self.prefit:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)

        X = check_array(X)

        if self.cv is not None:
            self.results_ = self._cv_feature_importances(
                X, y, groups=groups, **fit_params)
        else:
            self.results_ = self._non_cv_feature_importances(X, y)

        self.feature_importances_ = np.std(self.results_, axis=0)
        self.feature_importances_std_ = np.std(self.results_, axis=0)
        return self

    def _cv_feature_importances(self, X, y, groups=None, **fit_params):
        assert self.cv is not None
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        feature_importances = []  # type: List
        for train, test in cv.split(X, y, groups):
            est = clone(self.estimator).fit(X[train], y[train], **fit_params)
            score_func = partial(self.scorer_, est)
            feature_importances.extend(
                self._iter_feature_importances(score_func, X[test], y[test])
            )
        return feature_importances

    def _non_cv_feature_importances(self, X, y):
        score_func = partial(self.scorer_, self._fit_estimator)
        return list(self._iter_feature_importances(score_func, X, y))

    def _iter_feature_importances(self, score_func, X, y):
        for i in range(self.n_iter):
            yield score_decrease.get_feature_importances(
                score_func, X, y, random_state=self.rng_)

    # ============= Exposed methods of a wrapped estimator:

    def fit_transform(self, X, y, groups=None, **fit_params):
        self.fit(X, y, groups=groups, **fit_params)
        return self.transform(X)

    @if_delegate_has_method(delegate='_fit_estimator')
    def transform(self, X):
        return self._fit_estimator.transform(X)

    @if_delegate_has_method(delegate='_fit_estimator')
    def score(self, X, y=None, *args, **kwargs):
        return self._fit_estimator.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='_fit_estimator')
    def predict(self, X):
        return self._fit_estimator.predict(X)

    @if_delegate_has_method(delegate='_fit_estimator')
    def predict_proba(self, X):
        return self._fit_estimator.predict_proba(X)

    @if_delegate_has_method(delegate='_fit_estimator')
    def predict_log_proba(self, X):
        return self._fit_estimator.predict_log_proba(X)

    @if_delegate_has_method(delegate='_fit_estimator')
    def decision_function(self, X):
        return self._fit_estimator.decision_function(X)

    @property
    def _fit_estimator(self):
        if self.prefit:
            return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self._fit_estimator.classes_
