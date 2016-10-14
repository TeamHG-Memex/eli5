from eli5.formatters import format_html_styles


def test_render_styles():
    styles = format_html_styles()
    assert styles.strip().startswith('<style')