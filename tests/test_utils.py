# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import integers

from eli5.utils import argsort_k_largest, argsort_k_smallest
from .utils import rnd_len_arrays


@given(rnd_len_arrays(np.float32, 0, 5), integers(1, 6))
def test_argsort_k_smallest(x, k):
    assume(len(x) >= k)
    assume(len(set(x)) == len(x))
    assume(not np.isnan(x).any())
    assert (np.argsort(x)[:k] == argsort_k_smallest(x, k)).all()


@given(rnd_len_arrays(np.float32, 0, 5), integers(1, 6))
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
