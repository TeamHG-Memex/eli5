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


def get_value_indices(names, lookups):
    """
    >>> get_value_indices(['foo', 'bar', 'baz'], ['bar', 'foo'])
    [1, 0]
    >>> get_value_indices(['foo', 'bar', 'baz'], ['spam'])
    Traceback (most recent call last):
    ...
    KeyError: 'spam'
    """
    positions = {name: idx for idx, name in enumerate(names)}
    return [positions[name] for name in lookups]


def get_display_names(original_names=None, target_names=None, target_order=None):
    """
    Return a list of (class_id, display_name) tuples.

    Provide display names:
    >>> get_display_names([0, 2], target_names=['foo', 'bar'])
    [(0, 'foo'), (1, 'bar')]

    Change order of labels:
    >>> get_display_names(['x', 'y'], target_order=['y', 'x'])
    [(1, 'y'), (0, 'x')]

    Provide display names, choose only a subset of labels:
    >>> get_display_names([0, 2], target_names=['foo', 'bar'], target_order=[2])
    [(1, 'bar')]

    target_names can be a dictionary with {old_name: new_name} labels:
    >>> get_display_names(['x', 'y'], target_order=['y', 'x'],
    ...                   target_names={'x': 'X'})
    [(1, 'y'), (0, 'X')]

    """
    if isinstance(target_names, (list, tuple, np.ndarray)):
        if original_names is not None:
            if len(target_names) != len(original_names):
                raise ValueError("target_names must have the same length as "
                                 "original names: expected {}, got {}".format(
                                     len(original_names), len(target_names)
                                 ))
        display_names = target_names
    else:
        display_names = original_names

    if target_order is None:
        target_order = original_names

    class_indices = get_value_indices(original_names, target_order)
    names = [display_names[i] for i in class_indices]
    if isinstance(target_names, dict):
        names = [target_names.get(name, name) for name in names]
    return list(zip(class_indices, names))
