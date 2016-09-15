# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .utils import argsort_k_largest


def get_top_feature_weights(feature_names, coef, k):
    """
    Return a (positive_features, negative_features, is_truncated) tuple.
    """
    indices = argsort_k_largest(np.abs(coef), k)
    names, values = feature_names[indices], coef[indices]
    pos = [(name, value) for (name, value) in zip(names, values) if value > 0]
    neg = [(name, value) for (name, value) in zip(names, values) if value < 0]
    truncated = (len(pos) + len(neg)) != (coef != 0).sum()
    return pos, neg, truncated


# def get_top_positive_features(feature_names, coef, k):
#     indices = argsort_k_largest(coef, k)
#     names, values = feature_names[indices], coef[indices]
#     return [(name, value) for (name, value) in zip(names, values)
#             if value > 0]
#
#
# def get_top_negative_features(feature_names, coef, k):
#     indices = argsort_k_smallest(coef, k)
#     names, values = feature_names[indices], coef[indices]
#     return [(name, value) for (name, value) in zip(names, values)
#             if value < 0]
