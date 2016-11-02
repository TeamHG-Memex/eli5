import re

from eli5.base import WeightedSpans
from eli5.formatters import format_html_styles, FormattedFeatureName
from eli5.formatters.html import (
    _format_unhashed_feature, render_weighted_spans, _format_single_feature,
    _format_feature, _remaining_weight_color, _weight_color)


def test_render_styles():
    styles = format_html_styles()
    assert styles.strip().startswith('<style')


def test_format_unhashed_feature():
    assert _format_unhashed_feature([], 1) == ''
    assert _format_unhashed_feature([{'name': 'foo', 'sign': 1}], 1) == 'foo'
    assert _format_unhashed_feature([{'name': 'foo', 'sign': -1}], 1) == '(-)foo'
    assert _format_unhashed_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1}
        ], 1) == 'foo <span title="(-)bar">&hellip;</span>'
    assert _format_unhashed_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1},
        {'name': 'boo', 'sign': 1},
    ], 1) == 'foo <span title="(-)bar\nboo">&hellip;</span>'


def test_format_formatted_feature():
    assert _format_feature(FormattedFeatureName('a b'), 1) == 'a b'
    assert _format_feature('a b', 1) != 'a b'
    assert _format_feature('a b', 1) == _format_single_feature('a b', 1)


def test_format_single_feature():
    assert _format_single_feature('a', 1) == 'a'
    assert _format_single_feature('<>', 1) == '&lt;&gt;'
    assert _format_single_feature('aa bb', 1) == (
        'aa'
        '<span '
        'style="background-color: hsl(120, 80%, 70%); margin: 0 0.1em 0 0.1em" '
        'title="A space symbol">'
        '&emsp;'
        '</span>'
        'bb')
    assert _format_single_feature('  aa bb ', -1) == (
        '<span '
        'style="background-color: hsl(0, 80%, 70%); margin: 0 0.1em 0 0" '
        'title="2 space symbols">'
        '&emsp;'
        '&emsp;'
        '</span>'
        'aa'
        '<span '
        'style="background-color: hsl(0, 80%, 70%); margin: 0 0.1em 0 0.1em" '
        'title="A space symbol">'
        '&emsp;'
        '</span>'
        'bb'
        '<span '
        'style="background-color: hsl(0, 80%, 70%); margin: 0 0 0 0.1em" '
        'title="A space symbol">'
        '&emsp;'
        '</span>'
    )


def test_render_weighted_spans_word():
    weighted_spans = WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 0.2),
            ('tree', [(23, 27)], -0.6),
            ('leaning lemon', [(9, 16), (17, 22)], 0.5),
            ('lemon tree', [(17, 22), (23, 27)], 0.8)],
    )
    s = render_weighted_spans(weighted_spans)
    assert s.startswith(
        '<span style="opacity: 0.80">i</span>'
        '<span style="opacity: 0.80"> </span>'
        '<span'
        ' style="background-color: hsl(120, 100.00%, 89.21%); opacity: 0.83"'
        ' title="0.200">s</span>'
    )
    s_without_styles = re.sub('style=".*?"', '', s)
    assert s_without_styles == (
         '<span >i</span>'
         '<span > </span>'
         '<span  title="0.200">s</span>'
         '<span  title="0.200">e</span>'
         '<span  title="0.200">e</span>'
         '<span >:</span>'
         '<span > </span>'
         '<span >a</span>'
         '<span > </span>'
         '<span  title="0.500">l</span>'
         '<span  title="0.500">e</span>'
         '<span  title="0.500">a</span>'
         '<span  title="0.500">n</span>'
         '<span  title="0.500">i</span>'
         '<span  title="0.500">n</span>'
         '<span  title="0.500">g</span>'
         '<span > </span>'
         '<span  title="1.300">l</span>'
         '<span  title="1.300">e</span>'
         '<span  title="1.300">m</span>'
         '<span  title="1.300">o</span>'
         '<span  title="1.300">n</span>'
         '<span > </span>'
         '<span  title="0.200">t</span>'
         '<span  title="0.200">r</span>'
         '<span  title="0.200">e</span>'
         '<span  title="0.200">e</span>'
    )


def test_render_weighted_spans_char():
    weighted_spans = WeightedSpans(
        analyzer='char',
        document='see',
        weighted_spans=[
            ('se', [(0, 2)], 0.2),
            ('ee', [(1, 3)], 0.1),
            ],
    )
    s = render_weighted_spans(weighted_spans)
    assert s == (
        '<span'
        ' style="background-color: hsl(120, 100.00%, 69.88%); opacity: 0.93"'
        ' title="0.100">s</span>'
        '<span'
        ' style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00"'
        ' title="0.150">e</span>'
        '<span'
        ' style="background-color: hsl(120, 100.00%, 81.46%); opacity: 0.87"'
        ' title="0.050">e</span>'
    )


def test_override_preserve_density():
    weighted_spans = WeightedSpans(
        analyzer='char',
        document='see',
        weighted_spans=[
            ('se', [(0, 2)], 0.2),
            ('ee', [(1, 3)], 0.1),
        ],
    )
    s = render_weighted_spans(weighted_spans, preserve_density=False)
    assert s.startswith(
        '<span '
        'style="background-color: hsl(120, 100.00%, 69.88%); opacity: 0.93" '
        'title="0.200">s</span>')


def test_remaining_weight_color():
    assert _remaining_weight_color([], 0, 'pos') == _weight_color(1, 1)
    assert _remaining_weight_color([], 2, 'neg') == _weight_color(-2, 2)
    assert _remaining_weight_color([('a', -1), ('b', -2)], 3, 'neg') == \
        _weight_color(-1, 3)
    assert _remaining_weight_color([('a', 1), ('b', 2)], 3, 'pos') == \
           _weight_color(1, 3)
