# -*- coding: utf-8 -*-
import cgi

import numpy as np
from jinja2 import Environment, PackageLoader


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
flt = template_env.filters
flt['render_weighted_spans'] = lambda x: render_weighted_spans(x)
flt['weight_color'] = lambda w, w_range: _weight_color(w, w_range)
flt['weight_range'] = lambda w: _weight_range(w)


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
    return max(abs(coef) for key in ['pos', 'neg']
               for _, coef in weights.get(key, []))
