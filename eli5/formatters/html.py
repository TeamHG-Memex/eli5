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
    template = template_env.get_template('explain.html')
    return template.render(
        include_styles=include_styles,
        force_weights=force_weights,
        **explanation)


def format_html_styles():
    return template_env.get_template('styles.html').render()


def render_weighted_spans(weighted_spans_data):
    """ Render text document with highlighted features.
    """
    doc = weighted_spans_data['document']
    weighted_spans = weighted_spans_data['weighted_spans']
    char_weights = np.zeros(len(doc))
    for spans, weight in weighted_spans:
        for start, end in spans:
            char_weights[start:end] += weight
    # TODO - can be much smarter, join spans at least
    # TODO - for longer documents, remove text without active features
    weight_range = max(abs(char_weights.min()), abs(char_weights.max()))
    not_found_weights = sorted(
        (feature, weight)
        for feature, weight in weighted_spans_data['not_found'].items()
        if not np.isclose(weight, 0.))
    hl_doc = []
    if not_found_weights:
        hl_doc.append(' '.join(_colorize(token, weight, weight_range)
                            for token, weight in not_found_weights))
    hl_doc.append(''.join(_colorize(token, weight, weight_range)
                       for token, weight in zip(doc, char_weights)))
    return ' '.join(hl_doc)


def _colorize(token, weight, weight_range):
    token = cgi.escape(token, quote=True)
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
    """ Return css color for given weight, were the max absolute weight
    is given by weight_range.
    """
    hue = 120 if weight > 0 else 0
    saturation = 1
    lightness = 1.0 - 0.4 * abs(weight) / weight_range
    alpha = 1.
    return 'hsla({}, {:.2%}, {:.2%}, {:.4f})'.format(
        hue, saturation, lightness, alpha)


def _weight_range(weights):
    return max([abs(coef) for key in ['pos', 'neg']
                for _, coef in weights.get(key, [])] or [0])


def _format_unhashed_feature(feature):
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
    if (isinstance(feature, list) and
            ('name' in x and 'sign' in x for x in feature)):
        return _format_unhashed_feature(feature)
    else:
        return html_escape(feature)


def html_escape(text):
    return cgi.escape(text, quote=True)
