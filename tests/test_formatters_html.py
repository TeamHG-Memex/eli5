from eli5.formatters import format_html_styles
from eli5.formatters.html import _format_unhashed_feature


def test_render_styles():
    styles = format_html_styles()
    assert styles.strip().startswith('<style')


def test_format_unhashed_feature():
    assert _format_unhashed_feature([]) == ''
    assert _format_unhashed_feature([{'name': 'foo', 'sign': 1}]) == 'foo'
    assert _format_unhashed_feature([{'name': 'foo', 'sign': -1}]) == '(-)foo'
    assert _format_unhashed_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1}
        ]) == 'foo <span title="(-)bar">&hellip;</span>'
    assert _format_unhashed_feature([
        {'name': 'foo', 'sign': 1},
        {'name': 'bar', 'sign': -1},
        {'name': 'boo', 'sign': 1},
    ]) == 'foo <span title="(-)bar\nboo">&hellip;</span>'
