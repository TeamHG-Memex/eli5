# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import List, Any

import numpy as np
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state, issparse
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import shuffle as _shuffle

from eli5.utils import vstack
from eli5.sklearn.utils import sklearn_version


def fit_proba(clf, X, y_proba, expand_factor=10, sample_weight=None,
              shuffle=True, random_state=None,
              **fit_params):
    """
    Fit classifier ``clf`` to return probabilities close to ``y_proba``.

    scikit-learn can't optimize cross-entropy directly if target
    probability values are not indicator vectors. As a workaround this function
    expands the dataset according to target probabilities.
    Use expand_factor=None to turn it off
    (e.g. if probability scores are 0/1 in a first place).
    """
    X, y, sample_weight = expanded_X_y_sample_weights(X, y_proba,
        expand_factor=expand_factor,
        sample_weight=sample_weight,
        shuffle=shuffle,
        random_state=random_state,
    )
    fit_params = with_sample_weight(clf, sample_weight, fit_params)
    clf.fit(X, y, **fit_params)
    return clf


def with_sample_weight(clf, sample_weight, fit_params):
    """
    Return fit_params with added "sample_weight" argument.
    Unlike `fit_params['sample_weight'] = sample_weight` it
    handles a case where ``clf`` is a pipeline.
    """
    param_name = _get_classifier_prefix(clf) + "sample_weight"
    params = {param_name: sample_weight}
    params.update(fit_params)
    return params


def fix_multiclass_predict_proba(y_proba,          # type: np.ndarray
                                 seen_classes,
                                 complete_classes
                                 ):
    # type: (...) -> np.ndarray
    """
    Add missing columns to predict_proba result.

    When a multiclass classifier is fit on a dataset which only contains
    a subset of possible classes its predict_proba result only has columns
    corresponding to seen classes. This function adds missing columns.
    """
    assert set(complete_classes) >= set(seen_classes)
    y_proba_fixed = np.zeros(
        shape=(y_proba.shape[0], len(complete_classes)),
        dtype=y_proba.dtype,
    )
    class_mapping = np.searchsorted(complete_classes, seen_classes)
    y_proba_fixed[:, class_mapping] = y_proba
    return y_proba_fixed


class _PipelinePatched(Pipeline):
    # Patch from https://github.com/scikit-learn/scikit-learn/pull/7723;
    # only needed for scikit-learn < 0.19.
    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, **score_params):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].score(Xt, y, **score_params)


def score_with_sample_weight(estimator, X, y=None, sample_weight=None):
    if sklearn_version() < '0.19':
        if isinstance(estimator, Pipeline) and sample_weight is not None:
            estimator = _PipelinePatched(estimator.steps)
    if sample_weight is None:
        return estimator.score(X, y)
    return estimator.score(X, y, sample_weight=sample_weight)


def expanded_X_y_sample_weights(X, y_proba, expand_factor=10,
                                sample_weight=None, shuffle=True,
                                random_state=None):
    """
    scikit-learn can't optimize cross-entropy directly if target
    probability values are not indicator vectors.
    As a workaround this function expands the dataset according to
    target probabilities. ``expand_factor=None`` means no dataset
    expansion.
    """
    rng = check_random_state(random_state)
    if expand_factor:
        if sample_weight is not None:
            X, y, sample_weight = zip(*expand_dataset(X, y_proba,
                                                      factor=expand_factor,
                                                      random_state=rng,
                                                      extra_arrays=[
                                                          sample_weight
                                                      ]))
        else:
            X, y = zip(*expand_dataset(X, y_proba,
                                       factor=expand_factor,
                                       random_state=rng))
    else:
        y = y_proba.argmax(axis=1)

    if isinstance(X, (list, tuple)) and len(X) and issparse(X[0]):
        X = vstack(X)

    if shuffle:
        if sample_weight is not None:
            X, y, sample_weight = _shuffle(X, y, sample_weight,
                                           random_state=rng)
        else:
            X, y = _shuffle(X, y, random_state=rng)
    return X, y, sample_weight


def expand_dataset(X, y_proba, factor=10, random_state=None, extra_arrays=None):
    """
    Convert a dataset with float multiclass probabilities to a dataset
    with indicator probabilities by duplicating X rows and sampling
    true labels.
    """
    rng = check_random_state(random_state)
    extra_arrays = extra_arrays or []
    n_classes = y_proba.shape[1]
    classes = np.arange(n_classes, dtype=int)
    for el in zip(X, y_proba, *extra_arrays):
        x, probs = el[0:2]
        rest = el[2:]
        for label in rng.choice(classes, size=factor, p=probs):
            yield (x, label) + rest


def rbf(distance, sigma=1.0):
    """
    Convert distance to similarity in [0, 1] range using RBF (Gaussian)
    kernel.
    """
    return np.exp(-distance ** 2 / (2 * sigma ** 2))


def _get_classifier_prefix(clf_or_pipeline):
    """
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import make_pipeline
    >>> _get_classifier_prefix(LogisticRegression())
    ''
    >>> pipe = make_pipeline(CountVectorizer(), LogisticRegression())
    >>> _get_classifier_prefix(pipe)
    'logisticregression__'
    """
    if not isinstance(clf_or_pipeline, Pipeline):
        return ''
    return clf_or_pipeline.steps[-1][0] + "__"


def mean_kl_divergence(y_proba_pred, y_proba_target,
                       sample_weight=None, eps=1e-9):
    kl_elementwise = entropy(y_proba_target.T, y_proba_pred.T + eps)
    return np.average(kl_elementwise, weights=sample_weight)

