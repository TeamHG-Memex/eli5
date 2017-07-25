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


CAVEATS_CV_NONE = """
Feature importances are computed on the same data as used for training, 
i.e. feature importances don't reflect importance of features for 
generalization.
"""

CAVEATS_CV = """
Feature importances are not computed for the final estimator; 
they are computed for a sequence of estimators trained and evaluated 
on train/test splits. So they tell you about importances of features 
for generalization, but not feature importances of a particular trained model.
"""

CAVEATS_PREFIT = """
If feature importances are computed on the same data as used for training, 
they don't reflect importance of features for generalization. Use a held-out
dataset if you want generalization feature importances.
"""


class ScoreDecreaseFeatureImportances(BaseEstimator, MetaEstimatorMixin):
    """Meta-transformer which exposes ``feature_importances_`` attribute
    based on average score decrease.

    ScoreDecreaseFeatureImportances instance can be used instead of
    its wrapped estimator, as it exposes all estimator's common methods like
    ``predict``.

    There are 3 main modes of operation:

    1. cv="prefit" (pre-fit estimator is passed). You can call
       ScoreDecreaseFeatureImportances.fit either with training data, or
       with a held-out dataset (in the latter case ``feature_importances_``
       would be importances of features for generalization). After the fitting
       ``feature_importances_`` attribute becomes available, but the estimator
       itself is not fit again. When cv="prefit", ``fit`` must be called
       directly, and ScoreDecreaseFeatureImportances cannot be used with
       ``cross_val_score``, ``GridSearchCV`` and similar utilities that clone
       the estimator.
    2. cv=None. In this case ``fit`` method fits the estimator and computes
       feature importances on the same data, i.e. feature importances
       don't reflect importance of features for generalization.
    3. all other ``cv`` values. ``fit`` method fits the estimator, but
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

    cv : int, cross-validation generator, iterable or "prefit"
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to disable cross-validation and compute feature importances
              on the same data as used for training.
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            - "prefit" string constant (default).

        If "prefit" is passed, it is assumed that ``estimator`` has been
        fitted already and all data is used for computing feature importances.

    refit : bool
        Whether to fit the estimator on the whole data if cross-validation
        is used (default is False).

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
    def __init__(self, estimator, scoring=None, n_iter=5, random_state=None,
                 cv='prefit', refit=True):
        # type: (...) -> None
        if isinstance(cv, str) and cv != "prefit":
            raise ValueError("Invalid cv value: {!r}".format(cv))
        self.refit = refit
        self.estimator = estimator
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        self.rng_ = check_random_state(random_state)

    def fit(self, X, y, groups=None, **fit_params):
        # type: (...) -> ScoreDecreaseFeatureImportances
        """Compute ``feature_importances_`` attribute and optionally
        fit the base estimator.

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

        if self.cv != "prefit" and self.refit:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **fit_params)

        X = check_array(X)

        if self.cv not in (None, "prefit"):
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
        score_func = partial(self.scorer_, self.wrapped_estimator_)
        return list(self._iter_feature_importances(score_func, X, y))

    def _iter_feature_importances(self, score_func, X, y):
        for i in range(self.n_iter):
            yield score_decrease.get_feature_importances(
                score_func, X, y, random_state=self.rng_)

    @property
    def caveats_(self):
        if self.cv == 'prefit':
            return CAVEATS_PREFIT
        elif self.cv is None:
            return CAVEATS_CV_NONE
        return CAVEATS_CV

    # ============= Exposed methods of a wrapped estimator:

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def score(self, X, y=None, *args, **kwargs):
        return self.wrapped_estimator_.score(X, y, *args, **kwargs)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict(self, X):
        return self.wrapped_estimator_.predict(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_proba(self, X):
        return self.wrapped_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def predict_log_proba(self, X):
        return self.wrapped_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='wrapped_estimator_')
    def decision_function(self, X):
        return self.wrapped_estimator_.decision_function(X)

    @property
    def wrapped_estimator_(self):
        if self.cv == "prefit" or not self.refit:
            return self.estimator
        return self.estimator_

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        return self.wrapped_estimator_.classes_
