import attr
import pytest
import numpy as np

from eli5.base_utils import attrs, numpy_to_python


def test_attrs_with_default():

    @attrs
    class WithDefault(object):
        def __init__(self, x, y=1):
            self.x = x
            self.y = y

    x_attr, y_attr = attr.fields(WithDefault)
    assert x_attr.name == 'x'
    assert y_attr.name == 'y'
    assert x_attr.default is attr.NOTHING
    assert y_attr.default == 1

    assert WithDefault(1) == WithDefault(1)
    assert WithDefault(1, 1) != WithDefault(1, 2)


def test_attrs_without_default():

    @attrs
    class WithoutDefault(object):
        def __init__(self, x):
            self.x = x

    x_attr, = attr.fields(WithoutDefault)
    assert x_attr.name == 'x'
    assert x_attr.default is attr.NOTHING

    assert WithoutDefault(1) == WithoutDefault(1)
    assert WithoutDefault(1) != WithoutDefault(2)


def test_attrs_with_repr():

    @attrs
    class WithRepr(object):
        def __init__(self, x):
            self.x = x

        def __repr__(self):
            return 'foo'

    assert hash(WithRepr(1)) == hash(WithRepr(1))
    assert repr(WithRepr(2)) == 'foo'


def test_bad_init():

    @attrs
    class BadInit(object):
        def __init__(self, x):
            self._x = x

    with pytest.raises(AttributeError):
        BadInit(1)


def test_numpy_to_python():
    assert numpy_to_python({
        'x': np.int32(12),
        'y': [np.ones(2)],
    }) == {
        'x': 12,
        'y': [[1.0, 1.0]],
    }
