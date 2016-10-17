# -*- coding: utf-8 -*-
import numpy as np

from jinja2 import Environment, PackageLoader


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.filters['render_weighted_spans'] = \
    lambda x: render_weighted_spans(x)


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
    return ''.join(
        _colorize(char, weight, weight_range)
        for char, weight in zip(doc, char_weights))


def _colorize(char, weight, weight_range):
    if np.isclose(weight, 0.):
        return (
            '<span '
            'style="opacity: {opacity}"'
            '>{char}</span>'.format(
                opacity=_weight_opacity(weight, weight_range),
                char=char)
        )
    else:
        return (
            '<span '
            'style="background-color: {color}; opacity: {opacity}" '
            'title="{weight:.3f}"'
            '>{char}</span>'.format(
                color=_weight_color(weight, weight_range),
                opacity=_weight_opacity(weight, weight_range),
                weight=weight,
                char=char)
        )


def _weight_opacity(weight, weight_range):
    """ Return opacity value for given weight as a string.
    """
    min_opacity = 0.5
    rel_weight = abs(weight) / weight_range
    return '{:.2f}'.format(min_opacity + (1 - min_opacity) * rel_weight)


def _weight_color(weight, weight_range):
    """ Return css color for given weight, were the max absolute weight
    is given by weight_range.
    """
    hue = 120 if weight > 0 else 0
    saturation = 1
    lightness = 1.0 - 0.5 * (abs(weight) / weight_range)
    alpha = 1.
    return 'hsla({}, {:.0%}, {:.0%}, {:.2f})'.format(
        hue, saturation, lightness, alpha)
