# -*- coding: utf-8 -*-
from __future__ import absolute_import
import cgi
from collections import Counter
import copy

import numpy as np
from jinja2 import Environment, PackageLoader

from eli5 import _graphviz
from eli5.base import WeightedSpans, FeatureWeights
from .utils import format_signed, replace_spaces, should_highlight_spaces
from . import fields
from .features import FormattedFeatureName
from .trees import tree2text


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.globals.update(zip=zip, numpy=np)
template_env.filters.update(dict(
    weight_color=lambda w, w_range: format_hsl(weight_color_hsl(w, w_range)),
    remaining_weight_color=lambda ws, w_range, pos_neg:
        format_hsl(remaining_weight_color_hsl(ws, w_range, pos_neg)),
    format_feature=lambda f, w, hl: _format_feature(f, w, hl_spaces=hl),
    format_decision_tree=lambda tree: _format_decision_tree(tree),
))


def format_as_html(explanation, include_styles=True, force_weights=True,
                   show=fields.ALL, preserve_density=None,
                   highlight_spaces=None, horizontal_layout=True):
    """ Format explanation as html.
    Most styles are inline, but some are included separately in <style> tag,
    you can omit them by passing ``include_styles=False`` and call
    ``format_html_styles`` to render them separately (or just omit them).
    With ``force_weights=False``, weights will not be displayed in a table for
    predictions where it is possible to show feature weights highlighted
    in the document.
    If ``highlight_spaces`` is None (default), spaces will be highlighted in
    feature names only if there are any spaces at the start or at the end of the
    feature. Setting it to True forces space highlighting, and setting it to
    False turns it off.
    If ``horizontal_layout`` is True (default), multiclass classifier
    weights are laid out horizontally.
    """
    template = template_env.get_template('explain.html')
    if highlight_spaces is None:
        highlight_spaces = should_highlight_spaces(explanation)
    targets = explanation.targets or []
    if len(targets) == 1:
        horizontal_layout = False

    rendered_weighted_spans = (
        render_targets_weighted_spans(targets, preserve_density))
    weighted_spans_others = [
        merge_weighted_spans_others(t.weighted_spans) for t in targets]

    return template.render(
        include_styles=include_styles,
        force_weights=force_weights,
        target_table_styles='border-collapse: collapse; border: none; margin-top: 0em;',
        tr_styles='border: none;',
        td1_styles='padding: 0 1em 0 0.5em; text-align: right; border: none;',
        tdm_styles='padding: 0 0.5em 0 0.5em; text-align: center; border: none;',
        td2_styles='padding: 0 0.5em 0 0.5em; text-align: left; border: none;',
        horizontal_layout_table_styles=
        'border-collapse: collapse; border: none; margin-bottom: 1.5em;',
        horizontal_layout_td_styles=
        'padding: 0px; border: 1px solid black; vertical-align: top;',
        horizontal_layout_header_styles=
        'padding: 0.5em; border: 1px solid black; text-align: center;',
        show=show,
        expl=explanation,
        hl_spaces=highlight_spaces,
        horizontal_layout=horizontal_layout,
        any_weighted_spans=any(t.weighted_spans for t in targets),
        feat_imp_weight_range=max_or_0(
            abs(fw.weight) for fw in (explanation.feature_importances or [])),
        target_weight_range=max_or_0(
            get_weight_range(t.feature_weights) for t in targets),
        other_weight_range=max_or_0(
            get_weight_range(other)
            for other in weighted_spans_others if other),
        targets_with_weighted_spans=list(
            zip(targets, rendered_weighted_spans, weighted_spans_others)),
    )


def format_html_styles():
    """ Format just the styles,
    use with ``format_as_html(explanation, include_styles=False)``.
    """
    return template_env.get_template('styles.html').render()


def render_targets_weighted_spans(targets, preserve_density):
    spans_char_weights = [
        [get_char_weights(ws, preserve_density=preserve_density)
         for ws in t.weighted_spans] if t.weighted_spans else None
        for t in targets]
    # TODO - comment
    max_idx = max_or_0(len(ch_w or []) for ch_w in spans_char_weights)
    spans_weight_ranges = [max_or_0(
        abs(x) for char_weights in spans_char_weights
        for x in (char_weights[idx] if char_weights is not None else []))
                           for idx in range(max_idx)]
    return [
        '<br/>'.join(
            '{}{}'.format(
                '<b>{}:</b> '.format(ws.vec_name) if ws.vec_name else '',
                render_weighted_spans(ws.document, ch_w, w_range))
            for ws, ch_w, w_range in zip(
                t.weighted_spans, char_weights, spans_weight_ranges))
        if t.weighted_spans else None
        for t, char_weights in zip(targets, spans_char_weights)]


def merge_weighted_spans_others(weighted_spans):
    if not weighted_spans:
        return None
    if len(weighted_spans) == 1:
        return weighted_spans[0].other
    return FeatureWeights(
        pos=[_renamed(fw, ws) for ws in weighted_spans
             for fw in ws.other.pos],
        neg=[_renamed(fw, ws) for ws in weighted_spans
             for fw in ws.other.neg],
        # All should be the same, so min is fine
        pos_remaining=min(ws.other.pos_remaining for ws in weighted_spans),
        neg_remaining=min(ws.other.neg_remaining for ws in weighted_spans),
    )


def _renamed(fw, ws):
    if not ws.vec_name:
        return fw
    fw = copy.copy(fw)
    renamed = lambda x: '{}: {}'.format(ws.vec_name, x)
    if isinstance(fw.feature, FormattedFeatureName):
        fw.feature = FormattedFeatureName(renamed(fw.feature.value))
    elif isinstance(fw.feature, list):
        fw.feature = [
            {'name': renamed(x['name']), 'sing': x['sign']} for x in fw.feature]
    else:
        fw.feature = renamed(fw.feature)
    return fw


def get_char_weights(weighted_spans_data, preserve_density=None):
    """ Return character weights for a text document with highlighted features.
    If preserve_density is True, then color for longer fragments will be
    less intensive than for shorter fragments, so that "sum" of intensities
    will correspond to feature weight.
    If preserve_density is None, then it's value is chosen depending on
    analyzer kind: it is preserved for "char" and "char_wb" analyzers,
    and not preserved for "word" analyzers.
    """
    if preserve_density is None:
        preserve_density = weighted_spans_data.analyzer.startswith('char')
    weighted_spans = weighted_spans_data.weighted_spans
    char_weights = np.zeros(len(weighted_spans_data.document))
    feature_counts = Counter(f for f, _, _ in weighted_spans)
    for feature, spans, weight in weighted_spans:
        for start, end in spans:
            if preserve_density:
                weight /= (end - start)
            weight /= feature_counts[feature]
            char_weights[start:end] += weight
    return char_weights


def render_weighted_spans(doc, char_weights, weight_range):
    # TODO - can be much smarter, join spans at least
    # TODO - for longer documents, remove text without active features
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
                color=format_hsl(
                    weight_color_hsl(weight, weight_range, min_lightness=0.6)),
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


def weight_color_hsl(weight, weight_range, min_lightness=0.8):
    """ Return HSL color components for given weight,
    where the max absolute weight is given by weight_range.
    """
    hue = _hue(weight)
    saturation = 1
    rel_weight = (abs(weight) / weight_range) ** 0.7
    lightness = 1.0 - (1 - min_lightness) * rel_weight
    return hue, saturation, lightness


def format_hsl(hsl_color):
    """ Format hsl color as css color string.
    """
    hue, saturation, lightness = hsl_color
    return 'hsl({}, {:.2%}, {:.2%})'.format(hue, saturation, lightness)


def _hue(weight):
    return 120 if weight > 0 else 0


def get_weight_range(weights):
    """ Max absolute feature for pos and neg weights.
    """
    return max_or_0(abs(fw.weight) for lst in [weights.pos, weights.neg]
                    for fw in lst or [])


def remaining_weight_color_hsl(ws, weight_range, pos_neg):
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
        weight = min((fw.weight for fw in ws), key=abs)
    return weight_color_hsl(weight, weight_range)


def _format_unhashed_feature(feature, weight, hl_spaces):
    """ Format unhashed feature: show first (most probable) candidate,
    display other candidates in title attribute.
    """
    if not feature:
        return ''
    else:
        first, rest = feature[0], feature[1:]
        html = format_signed(
            first, lambda x: _format_single_feature(x, weight, hl_spaces))
        if rest:
            html += ' <span title="{}">&hellip;</span>'.format(
                '\n'.join(html_escape(format_signed(f)) for f in rest))
        return html


def _format_feature(feature, weight, hl_spaces):
    """ Format any feature.
    """
    if isinstance(feature, FormattedFeatureName):
        return feature.format()
    elif (isinstance(feature, list) and
            all('name' in x and 'sign' in x for x in feature)):
        return _format_unhashed_feature(feature, weight, hl_spaces=hl_spaces)
    else:
        return _format_single_feature(feature, weight, hl_spaces=hl_spaces)


def _format_single_feature(feature, weight, hl_spaces):
    feature = html_escape(feature)
    if not hl_spaces:
        return feature

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

    return replace_spaces(feature, replacer)


def _format_decision_tree(treedict):
    if treedict.graphviz and _graphviz.is_supported():
        return _graphviz.dot2svg(treedict.graphviz)
    else:
        return tree2text(treedict)


def html_escape(text):
    return cgi.escape(text, quote=True)


def max_or_0(it):
    lst = list(it)
    return max(lst) if lst else 0
