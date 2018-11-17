# -*- coding: utf-8 -*-
from __future__ import absolute_import
from itertools import groupby
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
from jinja2 import Environment, PackageLoader  # type: ignore

from eli5 import _graphviz
from eli5.base import (Explanation, TargetExplanation, FeatureWeights,
                       FeatureWeight)
from eli5.utils import max_or_0
from .utils import (
    format_signed, format_value, format_weight, has_any_values_for_weights,
    replace_spaces, should_highlight_spaces)
from . import fields
from .features import FormattedFeatureName
from .trees import tree2text
from .text_helpers import prepare_weighted_spans, PreparedWeightedSpans


template_env = Environment(
    loader=PackageLoader('eli5', 'templates'),
    extensions=['jinja2.ext.with_'])
template_env.globals.update(dict(zip=zip, numpy=np))
template_env.filters.update(dict(
    weight_color=lambda w, w_range: format_hsl(weight_color_hsl(w, w_range)),
    remaining_weight_color=lambda ws, w_range, pos_neg:
        format_hsl(remaining_weight_color_hsl(ws, w_range, pos_neg)),
    format_feature=lambda f, w, hl: _format_feature(f, w, hl_spaces=hl),
    format_value=format_value,
    format_weight=format_weight,
    format_decision_tree=lambda tree: _format_decision_tree(tree),
))


def format_as_html(explanation,  # type: Explanation
                   include_styles=True,  # type: bool
                   force_weights=True,  # type: bool
                   show=fields.ALL,
                   preserve_density=None,  # type: Optional[bool]
                   highlight_spaces=None,  # type: Optional[bool]
                   horizontal_layout=True,  # type: bool
                   show_feature_values=False  # type: bool
                   ):
    # type: (...) -> str
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
    If ``show_feature_values`` is True, feature values are shown if present.
    Default is False.
    """
    template = template_env.get_template('explain.html')
    if highlight_spaces is None:
        highlight_spaces = should_highlight_spaces(explanation)
    targets = explanation.targets or []
    if len(targets) == 1:
        horizontal_layout = False
    explaining_prediction = has_any_values_for_weights(explanation)
    show_feature_values = show_feature_values and explaining_prediction

    rendered_weighted_spans = render_targets_weighted_spans(
        targets, preserve_density)
    weighted_spans_others = [
        t.weighted_spans.other if t.weighted_spans else None for t in targets]

    return template.render(
        include_styles=include_styles,
        force_weights=force_weights,
        target_table_styles=
        'border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;',
        tr_styles='border: none;',
        # Weight (th and td)
        td1_styles='padding: 0 1em 0 0.5em; text-align: right; border: none;',
        # N more positive/negative
        tdm_styles='padding: 0 0.5em 0 0.5em; text-align: center; border: none; '
                   'white-space: nowrap;',
        # Feature (th and td)
        td2_styles='padding: 0 0.5em 0 0.5em; text-align: left; border: none;',
        # Value (th and td)
        td3_styles='padding: 0 0.5em 0 1em; text-align: right; border: none;',
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
            abs(fw.weight) for fw in explanation.feature_importances.importances)
        if explanation.feature_importances else 0,
        target_weight_range=max_or_0(
            get_weight_range(t.feature_weights) for t in targets),
        other_weight_range=max_or_0(
            get_weight_range(other)
            for other in weighted_spans_others if other),
        targets_with_weighted_spans=list(
            zip(targets, rendered_weighted_spans, weighted_spans_others)),
        show_feature_values=show_feature_values,
        weights_table_span=3 if show_feature_values else 2,
        explaining_prediction=explaining_prediction,
        weight_help=html_escape(WEIGHT_HELP),
        contribution_help=html_escape(CONTRIBUTION_HELP),
    )


WEIGHT_HELP = '''\
Feature weights. Note that weights do not account for feature value scales,
so if feature values have different scales, features with highest weights
might not be the most important.\
'''.replace('\n', ' ')
CONTRIBUTION_HELP = '''\
Feature contribution already accounts for the feature value
(for linear models, contribution = weight * feature value), and the sum
of feature contributions is equal to the score or, for some classifiers,
to the probability. Feature values are shown if "show_feature_values" is True.\
'''.replace('\n', ' ')


def format_html_styles():
    # type: () -> str
    """ Format just the styles,
    use with ``format_as_html(explanation, include_styles=False)``.
    """
    return template_env.get_template('styles.html').render()


def render_targets_weighted_spans(
        targets,  # type: List[TargetExplanation]
        preserve_density,  # type: Optional[bool]
    ):
    # type: (...) -> List[Optional[str]]
    """ Return a list of rendered weighted spans for targets.
    Function must accept a list in order to select consistent weight
    ranges across all targets.
    """
    prepared_weighted_spans = prepare_weighted_spans(
        targets, preserve_density)

    def _fmt_pws(pws):
        # type: (PreparedWeightedSpans) -> str
        name = ('<b>{}:</b> '.format(pws.doc_weighted_spans.vec_name)
                if pws.doc_weighted_spans.vec_name else '')
        return '{}{}'.format(name, render_weighted_spans(pws))

    def _fmt_pws_list(pws_lst):
        # type: (List[PreparedWeightedSpans]) -> str
        return '<br/>'.join(_fmt_pws(pws) for pws in pws_lst)

    return [_fmt_pws_list(pws_lst) if pws_lst else None
            for pws_lst in prepared_weighted_spans]


def render_weighted_spans(pws):
    # type: (PreparedWeightedSpans) -> str
    # TODO - for longer documents, an option to remove text
    # without active features
    return ''.join(
        _colorize(''.join(t for t, _ in tokens_weights),
                  weight,
                  pws.weight_range)
        for weight, tokens_weights in groupby(
            zip(pws.doc_weighted_spans.document, pws.char_weights),
            key=lambda x: x[1]))


def _colorize(token,  # type: str
              weight,  # type: float
              weight_range,  # type: float
              ):
    # type: (...) -> str
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
    # type: (float, float) -> str
    """ Return opacity value for given weight as a string.
    """
    min_opacity = 0.8
    if np.isclose(weight, 0) and np.isclose(weight_range, 0):
        rel_weight = 0.0
    else:
        rel_weight = abs(weight) / weight_range
    return '{:.2f}'.format(min_opacity + (1 - min_opacity) * rel_weight)


_HSL_COLOR = Tuple[float, float, float]


def weight_color_hsl(weight, weight_range, min_lightness=0.8):
    # type: (float, float, float) -> _HSL_COLOR
    """ Return HSL color components for given weight,
    where the max absolute weight is given by weight_range.
    """
    hue = _hue(weight)
    saturation = 1
    rel_weight = (abs(weight) / weight_range) ** 0.7
    lightness = 1.0 - (1 - min_lightness) * rel_weight
    return hue, saturation, lightness


def format_hsl(hsl_color):
    # type: (_HSL_COLOR) -> str
    """ Format hsl color as css color string.
    """
    hue, saturation, lightness = hsl_color
    return 'hsl({}, {:.2%}, {:.2%})'.format(hue, saturation, lightness)


def _hue(weight):
    # type: (float) -> float
    return 120 if weight > 0 else 0


def get_weight_range(weights):
    # type: (FeatureWeights) -> float
    """ Max absolute feature for pos and neg weights.
    """
    return max_or_0(abs(fw.weight)
                    for lst in [weights.pos, weights.neg]
                    for fw in lst or [])


def remaining_weight_color_hsl(
        ws,  # type: List[FeatureWeight]
        weight_range,  # type: float
        pos_neg,  # type: str
    ):
    # type: (...) -> _HSL_COLOR
    """ Color for "remaining" row.
    Handles a number of edge cases: if there are no weights in ws or weight_range
    is zero, assume the worst (most intensive positive or negative color).
    """
    sign = {'pos': 1.0, 'neg': -1.0}[pos_neg]
    if not ws and not weight_range:
        weight = sign
        weight_range = 1.0
    elif not ws:
        weight = sign * weight_range
    else:
        weight = min((fw.weight for fw in ws), key=abs)
    return weight_color_hsl(weight, weight_range)


def _format_unhashed_feature(feature, weight, hl_spaces):
    # type: (...) -> str
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
    # type: (...) -> str
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
    # type: (str, float, bool) -> str
    feature = html_escape(feature)
    if not hl_spaces:
        return feature

    def replacer(n_spaces, side):
        # type: (int, str) -> str
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
    # type: (...) -> str
    if treedict.graphviz and _graphviz.is_supported():
        return _graphviz.dot2svg(treedict.graphviz)
    else:
        return tree2text(treedict)


def html_escape(text):
    # type: (str) -> str
    try:
        from html import escape
    except ImportError:
        from cgi import escape  # type: ignore
    return escape(text, quote=True)
