# -*- coding: utf-8 -*-
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays


def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
    """ Generate numpy arrays of random length """
    lengths = integers(min_value=min_len, max_value=max_len)
    return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))
