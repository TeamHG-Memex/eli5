# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.pipeline import Pipeline


def fit_proba(clf, X, y_proba, expand_factor=10, sample_weight=None,
              **fit_params):
    """
    Fit classifier ``clf`` to return probabilities close to ``y_proba``.

    scikit-learn can't optimize cross-entropy directly if target
    probability values are not indicator vectors. As a workaround this function
    expands the dataset according to target probabilities.
    Use expand_factor=None to turn it off
    (e.g. if probability scores are 0/1 in a first place).
    """
    if expand_factor:
        if sample_weight is not None:
            X, y, sample_weight = zip(*expand_dataset(X, y_proba, expand_factor,
                                                      sample_weight))
        else:
            X, y = zip(*expand_dataset(X, y_proba, expand_factor))
    else:
        y = y_proba.argmax(axis=1)
    param_name = _get_classifier_prefix(clf) + "sample_weight"
    fit_params.setdefault(param_name, sample_weight)
    clf.fit(X, y, **fit_params)
    return clf


def expand_dataset(X, y_proba, factor=10, *arrays):
    """
    Convert a dataset with float multiclass probabilities to a dataset
    with indicator probabilities by duplicating X rows and sampling
    true labels.
    """
    n_classes = y_proba.shape[1]
    classes = np.arange(n_classes, dtype=int)
    for el in zip(X, y_proba, *arrays):
        x, probs = el[0:2]
        rest = el[2:]
        for label in np.random.choice(classes, size=factor, p=probs):
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
