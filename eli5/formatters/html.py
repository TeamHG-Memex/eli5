# -*- coding: utf-8 -*-
import cgi

import numpy as np
from jinja2 import Environment, PackageLoader

from .text import format_signed


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.filters.update(dict(
    render_weighted_spans=lambda x: render_weighted_spans(x),
    weight_color=lambda w, w_range: _weight_color(w, w_range),
    smallest_weight_color=lambda ws, w_range:
        _weight_color(min([coef for _, coef in ws] or [0], key=abs), w_range),
    weight_range=lambda w: _weight_range(w),
    fi_weight_range=lambda w: max([abs(x[1]) for x in w] or [0]),
    format_feature=lambda f: _format_feature(f),
))


def format_as_html(explanation, include_styles=True, force_weights=True):
    """ Format explanation as html.
    Most styles are inline, but some are included separately in <style> tag,
    you can omit them by passing ``include_styles=False`` and call
    ``format_html_styles`` to render them separately (or just omit them).
    With ``force_weights=False``, weights will not be displayed in a table for
    predictions where it is possible to show feature weights highlighted
    in the document.
    """
    template = template_env.get_template('explain.html')
    return template.render(
        include_styles=include_styles,
        force_weights=force_weights,
        table_styles='border-collapse: collapse; border: none;',
        tr_styles='border: none;',
        td1_styles='padding: 0 1em 0 0.5em; text-align: right; border: none;',
        tdm_styles='padding: 0 0.5em 0 0.5em; text-align: center; border: none;',
        td2_styles='padding: 0 0.5em 0 0.5em; text-align: left; border: none;',
        **explanation)


def format_html_styles():
    """ Format just the styles,
    use with ``format_as_html(explanation, include_styles=False)``.
    """
    return template_env.get_template('styles.html').render()


def render_weighted_spans(weighted_spans_data):
    """ Render text document with highlighted features.
    """
    doc = weighted_spans_data['document']
    weighted_spans = weighted_spans_data['weighted_spans']
    char_weights = np.zeros(len(doc))
    for _, spans, weight in weighted_spans:
        for start, end in spans:
            char_weights[start:end] += weight
    # TODO - can be much smarter, join spans at least
    # TODO - for longer documents, remove text without active features
    not_found_weights = sorted(
        (feature, weight)
        for feature, weight in weighted_spans_data['not_found'].items()
        if not np.isclose(weight, 0.))
    weight_range = max(abs(x) for x in char_weights)
    if not_found_weights:
        weight_range = max(weight_range,
                           max(abs(w) for _, w in not_found_weights))
    hl_doc = []
    if not_found_weights:
        hl_doc.append(' '.join(_colorize(token, weight, weight_range)
                               for token, weight in not_found_weights))
    hl_doc.append(''.join(_colorize(token, weight, weight_range)
                          for token, weight in zip(doc, char_weights)))
    return ' '.join(hl_doc)


def _colorize(token, weight, weight_range):
    """ Return token wrapped in a span with some styles
    (calculated from weight and weight_range) applied.
    """
    token = html_escape(token)
    if np.isclose(weight, 0.):
        return (
            '<span '
            'style="opacity: {opacity}"'
            '>{token}</span>'.format(
                opacity=_weight_opacity(weight, weight_range),
                token=token)
        )
    else:
        return (
            '<span '
            'style="background-color: {color}; opacity: {opacity}" '
            'title="{weight:.3f}"'
            '>{token}</span>'.format(
                color=_weight_color(weight, weight_range),
                opacity=_weight_opacity(weight, weight_range),
                weight=weight,
                token=token)
        )


def _weight_opacity(weight, weight_range):
    """ Return opacity value for given weight as a string.
    """
    min_opacity = 0.8
    rel_weight = abs(weight) / weight_range
    return '{:.2f}'.format(min_opacity + (1 - min_opacity) * rel_weight)


def _weight_color(weight, weight_range):
    """ Return css color for given weight, where the max absolute weight
    is given by weight_range.
    """
    hue = 120 if weight > 0 else 0
    saturation = 1
    min_lightness = 0.6
    lightness = 1.0 - (1 - min_lightness) * abs(weight) / weight_range
    return 'hsl({}, {:.2%}, {:.2%})'.format(hue, saturation, lightness)


def _weight_range(weights):
    """ Max absolute feature for pos and neg weights.
    """
    return max([abs(coef) for key in ['pos', 'neg']
                for _, coef in weights.get(key, [])] or [0])


def _format_unhashed_feature(feature):
    """ Format unhashed feature: show first (most probable) candidate,
    display other candidates in title attribute.
    """
    if not feature:
        return ''
    else:
        first, rest = feature[0], feature[1:]
        html = html_escape(format_signed(first))
        if rest:
            html += ' <span title="{}">&hellip;</span>'.format(
                '\n'.join(html_escape(format_signed(f)) for f in rest))
        return html


def _format_feature(feature):
    """ Format any feature.
    """
    if (isinstance(feature, list) and
            all('name' in x and 'sign' in x for x in feature)):
        return _format_unhashed_feature(feature)
    else:
        return html_escape(feature)


def html_escape(text):
    return cgi.escape(text, quote=True)
