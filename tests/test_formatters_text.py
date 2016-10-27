# -*- coding: utf-8 -*-

from eli5.formatters import FormattedFeatureName
from eli5.formatters.text import _format_feature, _SPACE


def test_format_formatted_feature():
    assert _format_feature(FormattedFeatureName('a b')) == 'a b'
    assert _format_feature('a b') == 'a{}b'.format(_SPACE)


def test_format_unhashed_feature():
    assert _format_feature([]) == ''
    assert _format_feature([{'name': 'foo', 'sign': 1}]) == 'foo'
    assert _format_feature([{'name': ' foo', 'sign': -1}]) == \
        '(-){}foo'.format(_SPACE)
    assert _format_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1}
    ]) == 'foo | (-)bar'
