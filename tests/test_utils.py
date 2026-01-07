# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sp
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import integers

from eli5.utils import (
    argsort_k_largest, argsort_k_largest_positive, argsort_k_smallest, vstack)
from .utils import rnd_len_arrays


@given(rnd_len_arrays(np.float32, 0, 5), integers(1, 6))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_argsort_k_smallest(x, k):
    assume(len(x) >= k)
    assume(len(set(x)) == len(x))
    assume(not np.isnan(x).any())
    assert (np.argsort(x)[:k] == argsort_k_smallest(x, k)).all()


@given(rnd_len_arrays(np.float32, 0, 5), integers(1, 6))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_argsort_k_largest(x, k):
    assume(len(x) >= k)
    assume(len(set(x)) == len(x))
    assume(not np.isnan(x).any())
    assert (np.argsort(x)[-k:][::-1] == argsort_k_largest(x, k)).all()


@given(rnd_len_arrays(np.float32, 0, 5))
def test_argsort_k_smallest_zero(x):
    assert len(argsort_k_smallest(x, 0)) == 0


@given(rnd_len_arrays(np.float32, 0, 5))
def test_argsort_k_smallest_None(x):
    assert len(argsort_k_smallest(x, None)) == len(x)


@given(rnd_len_arrays(np.float32, 0, 5))
def test_argsort_k_largest_zero(x):
    assert len(argsort_k_largest(x, 0)) == 0


@given(rnd_len_arrays(np.float32, 0, 5))
def test_argsort_k_largest_None(x):
    assert len(argsort_k_largest(x, None)) == len(x)


def test_argsort_k_largest_empty():
    x = np.array([0])
    empty = np.array([])
    assert _np_eq(x[argsort_k_largest(x, 0)], empty)
    assert _np_eq(x[argsort_k_largest_positive(x, None)], empty)


def test_argsort_k_largest_positive():
    assert _np_eq(argsort_k_largest_positive(np.array([1.0, 0.0, 2.0]), None),
                  np.array([2, 0]))
    assert _np_eq(argsort_k_largest_positive(np.array([1.0, 0.0, 2.0, 4.0]), 2),
                  np.array([3, 2]))


def test_vstack_empty():
    assert vstack([]).shape == (0,)


def test_vstack_dense():
    res = vstack([
        np.array([1, 2, 3]),
        np.array([4, 5, 6])
    ])
    assert res.shape == (2, 3)
    assert not sp.issparse(res)
    assert np.array_equal(res, np.array([[1, 2, 3], [4, 5, 6]]))


def test_vstack_sparse():
    res = vstack([
        sp.csr_matrix([1, 2, 3]),
        sp.csr_matrix([4, 5, 6]),
    ])
    assert res.shape == (2, 3)
    assert sp.issparse(res)
    assert np.array_equal(res.todense(), np.array([[1, 2, 3], [4, 5, 6]]))


def _np_eq(x, y):
    return x.shape == y.shape and np.allclose(x, y)
