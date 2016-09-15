# -*- coding: utf-8 -*-
import numpy as np


def argsort_k_largest(x, k):
    """ Return ``k`` indices of largest values. """
    if k == len(x):
        return np.argsort(x)[::-1]
    indices = np.argpartition(x, -k)[-k:]
    values = x[indices]
    return indices[np.argsort(-values)]


def argsort_k_smallest(x, k):
    """ Return ``k`` indices of smallest values. """
    if k == len(x):
        return np.argsort(x)
    indices = np.argpartition(x, k)[:k]
    values = x[indices]
    return indices[np.argsort(values)]
