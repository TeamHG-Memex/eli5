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
        c if np.isclose(weight, 0.) else
        '<span '
        'style="background-color: {color}" '
        'title="{weight:.3f}"'
        '>{c}</span>'.format(
            color=_weight_color(weight, weight_range),
            weight=weight,
            c=c)
        for c, weight in zip(doc, char_weights))


def _weight_color(weight, weight_range):
    """ Return css color for given weight, were the max absolute weight
    is given by weight_range.
    """
    # TODO - maybe there are better solutions for this in matplotlib
    alpha = (abs(weight) / weight_range) ** 1.5
    h, l = 255, 150
    if weight > 0:
        rgb = (l, h, l)
    else:
        rgb = (h, l, l)
    rbga = rgb + (alpha,)
    return 'rgba{}'.format(rbga)
