import attr
import pytest

from eli5.base_utils import attrs


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
    """ Check that if __repr__ or __hash__ is defined, attrs does not override
    them (there is a special logic for it in @attrs).
    """

    @attrs
    class WithRepr(object):
        def __init__(self, x):
            self.x = x

        def __repr__(self):
            return 'foo'

        def __hash__(self):
            return 1

    assert WithRepr(1) == WithRepr(1) != WithRepr(2)
    assert repr(WithRepr(2)) == 'foo'
    assert hash(WithRepr(2)) == 1


def test_bad_init():
    """ Constructor argument names must match attribute names.
    """

    @attrs
    class BadInit(object):
        def __init__(self, x):
            self._x = x

    with pytest.raises(AttributeError):
        BadInit(1)
