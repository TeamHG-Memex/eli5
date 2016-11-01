# -*- coding: utf-8 -*-
from __future__ import absolute_import
import cgi
from collections import Counter

import numpy as np
from jinja2 import Environment, PackageLoader

from eli5._dot import dot2svg
from .utils import format_signed, replace_spaces
from . import fields
from .features import FormattedFeatureName
from .trees import tree2text


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.filters.update(dict(
    render_weighted_spans=lambda x, pd: render_weighted_spans(x, pd),
    weight_color=lambda w, w_range: _weight_color(w, w_range),
    remaining_weight_color=lambda ws, w_range, pos_neg:
        _remaining_weight_color(ws, w_range, pos_neg),
    weight_range=lambda w: _weight_range(w),
    fi_weight_range=lambda w: max([abs(x[1]) for x in w] or [0]),
    format_feature=lambda f, w: _format_feature(f, w),
    format_decision_tree=lambda tree: _format_decision_tree(tree),
))


def format_as_html(explanation, include_styles=True, force_weights=True,
                   show=fields.ALL, preserve_density=None):
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
        preserve_density=preserve_density,
        table_styles='border-collapse: collapse; border: none;',
        tr_styles='border: none;',
        td1_styles='padding: 0 1em 0 0.5em; text-align: right; border: none;',
        tdm_styles='padding: 0 0.5em 0 0.5em; text-align: center; border: none;',
        td2_styles='padding: 0 0.5em 0 0.5em; text-align: left; border: none;',
        show=show,
        **explanation)


def format_html_styles():
    """ Format just the styles,
    use with ``format_as_html(explanation, include_styles=False)``.
    """
    return template_env.get_template('styles.html').render()


def render_weighted_spans(weighted_spans_data, preserve_density=None):
    """ Render text document with highlighted features.
    If preserve_density is True, then color for longer fragments will be
    less intensive than for shorter fragments, so that "sum" of intensities
    will correspond to feature weight.
    If preserve_density is None, then it's value is chosen depending on
    analyzer kind: it is preserved for "char" and "char_wb" analyzers,
    and not preserved for "word" analyzers.
    """
    if preserve_density is None:
        preserve_density = weighted_spans_data['analyzer'].startswith('char')
    doc = weighted_spans_data['document']
    weighted_spans = weighted_spans_data['weighted_spans']
    char_weights = np.zeros(len(doc))
    feature_counts = Counter(f for f, _, _ in weighted_spans)
    for feature, spans, weight in weighted_spans:
        for start, end in spans:
            if preserve_density:
                weight /= (end - start)
            weight /= feature_counts[feature]
            char_weights[start:end] += weight
    # TODO - can be much smarter, join spans at least
    # TODO - for longer documents, remove text without active features
    weight_range = max(abs(x) for x in char_weights)
    return ''.join(_colorize(token, weight, weight_range)
                   for token, weight in zip(doc, char_weights))


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
                color=_weight_color(weight, weight_range, min_lightness=0.6),
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


def _weight_color(weight, weight_range, min_lightness=0.8):
    """ Return css color for given weight, where the max absolute weight
    is given by weight_range.
    """
    hue = _hue(weight)
    saturation = 1
    rel_weight = (abs(weight) / weight_range) ** 0.7
    lightness = 1.0 - (1 - min_lightness) * rel_weight
    return 'hsl({}, {:.2%}, {:.2%})'.format(hue, saturation, lightness)


def _hue(weight):
    return 120 if weight > 0 else 0


def _weight_range(weights):
    """ Max absolute feature for pos and neg weights.
    """
    return max([abs(coef) for key in ['pos', 'neg']
                for _, coef in weights.get(key, [])] or [0])


def _remaining_weight_color(ws, weight_range, pos_neg):
    """ Color for "remaining" row.
    Handles a number of edge cases: if there are no weights in ws or weight_range
    is zero, assume the worst (most intensive positive or negative color).
    """
    sign = {'pos': 1, 'neg': -1}[pos_neg]
    if not ws and not weight_range:
        weight = sign
        weight_range = 1
    elif not ws:
        weight = sign * weight_range
    else:
        weight = min((coef for _, coef in ws), key=abs)
    return _weight_color(weight, weight_range)


def _format_unhashed_feature(feature, weight):
    """ Format unhashed feature: show first (most probable) candidate,
    display other candidates in title attribute.
    """
    if not feature:
        return ''
    else:
        first, rest = feature[0], feature[1:]
        html = format_signed(first, lambda x: _format_single_feature(x, weight))
        if rest:
            html += ' <span title="{}">&hellip;</span>'.format(
                '\n'.join(html_escape(format_signed(f)) for f in rest))
        return html


def _format_feature(feature, weight):
    """ Format any feature.
    """
    if isinstance(feature, FormattedFeatureName):
        return feature.format()
    elif (isinstance(feature, list) and
            all('name' in x and 'sign' in x for x in feature)):
        return _format_unhashed_feature(feature, weight)
    else:
        return _format_single_feature(feature, weight)


def _format_single_feature(feature, weight):

    def replacer(n_spaces, side):
        m = '0.1em'
        margins = {'left': (m, 0), 'right': (0, m), 'center': (m, m)}[side]
        style = '; '.join([
            'background-color: hsl({}, 80%, 70%)'.format(_hue(weight)),
            'margin: 0 {} 0 {}'.format(*margins),
        ])
        return '<span style="{style}" title="{title}">{spaces}</span>'.format(
            style=style,
            title='A space symbol' if n_spaces == 1 else
                  '{} space symbols'.format(n_spaces),
            spaces='&emsp;' * n_spaces)

    return replace_spaces(html_escape(feature), replacer)


def _format_decision_tree(treedict):
    if 'graphviz' in treedict:
        return dot2svg(treedict['graphviz'])
    else:
        return tree2text(treedict)


def html_escape(text):
    return cgi.escape(text, quote=True)
