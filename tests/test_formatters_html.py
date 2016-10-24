import re

from eli5.formatters import format_html_styles
from eli5.formatters.html import (
    _format_unhashed_feature, render_weighted_spans, _format_single_feature)


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


def test_render_weighted_spans():
    weighted_spans = {
        'document': 'i see: a leaning lemon tree',
        'weighted_spans': [
            ('see', [(2, 5)], 0.2),
            ('tree', [(23, 27)], -0.6),
            ('leaning lemon', [(9, 16), (17, 22)], 0.5),
            ('lemon tree', [(17, 22), (23, 27)], 0.8)],
        'not_found': {'<BIAS>': 0.5}
    }
    s = render_weighted_spans(weighted_spans)
    assert s.startswith(
        '<span'
        ' style="background-color: hsl(120, 100.00%, 79.51%); opacity: 0.88"'
        ' title="0.500">&lt;BIAS&gt;</span> '
        '<span style="opacity: 0.80">i</span>'
        '<span style="opacity: 0.80"> </span>'
        '<span'
        ' style="background-color: hsl(120, 100.00%, 89.21%); opacity: 0.83"'
        ' title="0.200">s</span>'
    )
    s_without_styles = re.sub('style=".*?"', '', s)
    assert s_without_styles == (
         '<span  title="0.500">&lt;BIAS&gt;</span> <span >i</span>'
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
