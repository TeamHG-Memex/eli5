"""
A sklearn-compatible object for computing score decrease-based
feature importances.
"""
from functools import partial

import numpy as np  # type: ignore
from sklearn.utils.metaestimators import if_delegate_has_method  # type: ignore
from sklearn.utils import check_array, check_random_state  # type: ignore
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone  # type: ignore
from sklearn.metrics.scorer import check_scoring  # type: ignore

from eli5 import score_decrease


class ScoreDecreaseFeatureImportances(BaseEstimator, MetaEstimatorMixin):
    """Meta-transformer which exposes feature_importances_ attribute
    based on average score decrease.

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

    Attributes
    ----------
    feature_importances_ : array
        Feature importances, computed as mean decrease of the score.

    feature_importances_std_ : array
        Standard deviations of feature importances.

    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``ScoreDecreaseFeatureImportances``, i.e when prefit is False.

    rng_ : numpy.random.RandomState
        random state
    """
    def __init__(self, estimator, prefit=True, scoring=None, n_iter=5,
                 random_state=None):
        self.estimator = estimator
        self.prefit = prefit
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)

    def fit(self, X, y, **fit_params):
        """Compute ``feature_importances_`` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

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
        score_func = partial(self.scorer_, self._fit_estimator)

        feature_importances = []
        for i in range(self.n_iter):
            feature_importances.append(score_decrease.get_feature_importances(
                score_func, X, y, random_state=self.rng_))
        self.feature_importances_ = np.std(feature_importances, axis=0)
        self.feature_importances_std_ = np.std(feature_importances, axis=0)
        return self

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
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
