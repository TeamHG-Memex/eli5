# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np  # type: ignore

from eli5.base import FeatureWeights, FeatureWeight
from .utils import argsort_k_largest_positive, argsort_k_smallest, mask
from .formatters.features import FormattedFeatureName


def _get_top_features(feature_names, coef, top, x, missing):
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
    * ``x`` is a vector of feature values, passed to FeatureWeight.value.
    * ``missing`` is a value that is considered "missing" in ``x``.
    """
    if isinstance(top, (list, tuple)):
        num_pos, num_neg = list(top)  # "list" is just for mypy
        pos = _get_top_positive_features(
            feature_names, coef, num_pos, x, missing)
        neg = _get_top_negative_features(
            feature_names, coef, num_neg, x, missing)
    else:
        pos, neg = _get_top_abs_features(
            feature_names, coef, top, x, missing)
    return pos, neg


def get_top_features(feature_names, coef, top, x=None, missing=np.nan,
                     filtered_weights=0):
    pos, neg = _get_top_features(feature_names, coef, top, x, missing)
    pos_coef = coef > 0
    neg_coef = coef < 0
    pos_remaining = pos_coef.sum() - len(pos)
    neg_remaining = neg_coef.sum() - len(neg)
    if filtered_weights:
        filtered_out = FeatureWeight(
            '<Filtered by feature_flt>', filtered_weights)
        # TODO - test!!!
        if filtered_weights > 0:
            for idx, fw in enumerate(pos):
                if fw.weight < filtered_weights:
                    pos.insert(idx, filtered_out)
                    break
        elif filtered_weights < 0:
            for idx, fw in enumerate(neg):
                if fw.weight > filtered_weights:
                    neg.insert(idx, filtered_out)
                    break
    # pos_sum = sum(w for name, w in pos or [['', 0]])
    # neg_sum = sum(w for name, w in neg or [['', 0]])
    return FeatureWeights(
         pos=pos,
         neg=neg,
         pos_remaining=pos_remaining,
         neg_remaining=neg_remaining,
         # pos_remaining_sum=coef[pos_coef].sum() - pos_sum,
         # neg_remaining_sum=coef[neg_coef].sum() - neg_sum,
    )


def _get_top_abs_features(feature_names, coef, k, x, missing):
    indices = argsort_k_largest_positive(np.abs(coef), k)
    features = _features(indices, feature_names, coef, x, missing)
    pos = [fw for fw in features if fw.weight > 0]
    neg = [fw for fw in features if fw.weight < 0]
    return pos, neg


def _get_top_positive_features(feature_names, coef, k, x, missing):
    indices = argsort_k_largest_positive(coef, k)
    return _features(indices, feature_names, coef, x, missing)


def _get_top_negative_features(feature_names, coef, k, x, missing):
    num_negative = (coef < 0).sum()
    k = num_negative if k is None else min(num_negative, k)
    indices = argsort_k_smallest(coef, k)
    return _features(indices, feature_names, coef, x, missing)


def _features(indices, feature_names, coef, x, missing):
    names = mask(feature_names, indices)
    weights = mask(coef, indices)
    if x is not None:
        values = mask(x, indices)
        if not np.isnan(missing):
            values[values == missing] = np.nan
        return [FeatureWeight(name, weight, value=value)
                for name, weight, value in zip(names, weights, values)]
    else:
        return [FeatureWeight(name, weight)
                for name, weight in zip(names, weights)]
