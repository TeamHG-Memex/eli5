# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse as sp


def argsort_k_largest(x, k):
    """ Return no more than ``k`` indices of largest values. """
    if k == 0:
        return np.array([])
    if k >= len(x):
        return np.argsort(x)[::-1]
    indices = np.argpartition(x, -k)[-k:]
    values = x[indices]
    return indices[np.argsort(-values)]


def argsort_k_smallest(x, k):
    """ Return no more than ``k`` indices of smallest values. """
    if k == 0:
        return np.array([])
    if k >= len(x):
        return np.argsort(x)
    indices = np.argpartition(x, k)[:k]
    values = x[indices]
    return indices[np.argsort(values)]


def mask(x, indices):
    """
    The same as x[indices], but return an empty array if indices are empty,
    instead of returning all x elements.
    """
    if not indices.shape[0]:
        return np.array([])
    return x[indices]


def vstack(blocks, format=None, dtype=None):
    if any(sp.issparse(b) for b in blocks):
        return sp.vstack(blocks, format=format, dtype=dtype)
    else:
        return np.vstack(blocks)
