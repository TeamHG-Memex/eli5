# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from eli5.base import FeatureWeights, FeatureWeight
from .utils import argsort_k_largest, argsort_k_smallest, mask


def _get_top_features(feature_names, coef, top):
    """
    Return a ``(pos, neg)`` tuple. ``pos`` and ``neg`` are lists of
    ``(name, value)`` tuples for features with positive and negative
    coefficients.

    Parameters:

    * ``feature_names`` - a vector of feature names;
    * ``coef`` - coefficient vector; coef.shape must be equal to
      feature_names.shape;
    * ``top`` can be either a number or a ``(num_pos, num_neg)`` tuple.
      If ``top`` is a number, ``top`` features with largest absolute
      coefficients are returned. If it is a ``(num_pos, num_neg)`` tuple,
      the function returns no more than ``num_pos`` positive features and
      no more than ``num_neg`` negative features. ``None`` value means
      'no limit'.
    """
    if isinstance(top, (list, tuple)):
        num_pos, num_neg = list(top)  # "list" is just for mypy
        pos = _get_top_positive_features(feature_names, coef, num_pos)
        neg = _get_top_negative_features(feature_names, coef, num_neg)
    else:
        pos, neg = _get_top_abs_features(feature_names, coef, top)
    return pos, neg


def get_top_features(feature_names, coef, top):
    pos, neg = _get_top_features(feature_names, coef, top)
    pos_coef = coef > 0
    neg_coef = coef < 0
    # pos_sum = sum(w for name, w in pos or [['', 0]])
    # neg_sum = sum(w for name, w in neg or [['', 0]])
    return FeatureWeights(
         pos=pos,
         neg=neg,
         pos_remaining=pos_coef.sum() - len(pos),
         neg_remaining=neg_coef.sum() - len(neg),
         # pos_remaining_sum=coef[pos_coef].sum() - pos_sum,
         # neg_remaining_sum=coef[neg_coef].sum() - neg_sum,
    )


def _get_top_abs_features(feature_names, coef, k):
    nnz = np.count_nonzero(coef)
    k = nnz if k is None else min(nnz, k)
    indices = argsort_k_largest(np.abs(coef), k)
    features = _features(indices, feature_names, coef)
    return _positive(features), _negative(features)


def _get_top_positive_features(feature_names, coef, k):
    num_positive = (coef > 0).sum()
    k = num_positive if k is None else min(num_positive, k)
    indices = argsort_k_largest(coef, k)
    return _features(indices, feature_names, coef)


def _get_top_negative_features(feature_names, coef, k):
    num_negative = (coef < 0).sum()
    k = num_negative if k is None else min(num_negative, k)
    indices = argsort_k_smallest(coef, k)
    return _features(indices, feature_names, coef)


def _positive(features):
    return [fw for fw in features if fw.weight > 0]


def _negative(features):
    return [fw for fw in features if fw.weight < 0]


def _features(indices, feature_names, coef):
    names = mask(feature_names, indices)
    values = mask(coef, indices)
    return [FeatureWeight(name, weight) for name, weight in zip(names, values)]


