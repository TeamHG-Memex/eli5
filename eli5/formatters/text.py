# -*- coding: utf-8 -*-
from __future__ import absolute_import
from itertools import chain
import six
from typing import List, Optional, Iterator

from eli5.base import Explanation, FeatureImportances
from . import fields
from .features import FormattedFeatureName
from .utils import (
    format_signed, format_value, format_weight, has_any_values_for_weights,
    replace_spaces, should_highlight_spaces, tabulate)
from .trees import tree2text


_PLUS_MINUS = "+-" if six.PY2 else "±"
_ELLIPSIS = '...' if six.PY2 else '…'
_SPACE = '_' if six.PY2 else '░'


def format_as_text(expl,  # type: Explanation
                   show=fields.ALL,
                   highlight_spaces=None,  # type: Optional[bool]
                   show_feature_values=False,  # type: bool
                   ):
    # type: (...) -> str
    """ Format explanation as text.

    Parameters
    ----------
    expl : eli5.base.Explanation
        Explanation returned by ``eli5.explain_weights`` or
        ``eli5.explain_prediction`` functions.

    highlight_spaces : bool or None, optional
        Whether to highlight spaces in feature names. This is useful if
        you work with text and have ngram features which may include spaces
        at left or right. Default is None, meaning that the value used
        is set automatically based on vectorizer and feature values.

    show_feature_values : bool
        When True, feature values are shown along with feature contributions.
        Default is False.

    show : List[str], optional
        List of sections to show. Allowed values:

        * 'targets' - per-target feature weights;
        * 'transition_features' - transition features of a CRF model;
        * 'feature_importances' - feature importances of a decision tree or
          an ensemble-based estimator;
        * 'decision_tree' - decision tree in a graphical form;
        * 'method' - a string with explanation method;
        * 'description' - description of explanation method and its caveats.

        ``eli5.formatters.fields`` provides constants that cover common cases:
        ``INFO`` (method and description), ``WEIGHTS`` (all the rest),
        and ``ALL`` (all).
    """
    lines = []  # type: List[str]

    if highlight_spaces is None:
        highlight_spaces = should_highlight_spaces(expl)

    if expl.error:  # always shown
        lines.extend(_error_lines(expl))

    explaining_prediction = has_any_values_for_weights(expl)
    show_feature_values = show_feature_values and explaining_prediction

    for key in show:
        if not getattr(expl, key, None):
            continue

        if key == 'method':
            lines.extend(_method_lines(expl))

        if key == 'description':
            lines.extend(_description_lines(expl))

        if key == 'transition_features':
            lines.extend(_transition_features_lines(expl))

        if key == 'targets':
            lines.extend(_targets_lines(
                expl,
                hl_spaces=highlight_spaces,
                show_feature_values=show_feature_values,
                explaining_prediction=explaining_prediction,
            ))

        if key == 'feature_importances':
            lines.extend(_feature_importances_lines(
                expl, hl_spaces=highlight_spaces))

        if key == 'decision_tree':
            lines.extend(_decision_tree_lines(expl))

    return '\n'.join(lines)


def _method_lines(explanation):
    # type: (Explanation) -> List[str]
    return ['Explained as: {}'.format(explanation.method)]


def _description_lines(explanation):
    # type: (Explanation) -> List[str]
    return [explanation.description or '']


def _error_lines(explanation):
    # type: (Explanation) -> List[str]
    return ['Error: {}'.format(explanation.error)]


def _feature_importances_lines(explanation, hl_spaces):
    # type: (Explanation, Optional[bool]) -> Iterator[str]
    max_width = 0
    assert explanation.feature_importances is not None
    for line in _fi_lines(explanation.feature_importances, hl_spaces):
        max_width = max(max_width, len(line))
        yield line
    if explanation.feature_importances.remaining:
        yield _format_remaining(
            explanation.feature_importances.remaining, kind='', width=max_width)


def _fi_lines(feature_importances, hl_spaces):
    # type: (FeatureImportances, Optional[bool]) -> Iterator[str]
    for fw in feature_importances.importances:
        featname = _format_feature(fw.feature, hl_spaces)
        if fw.std or fw.weight:
            w = u'{:0.4f}'.format(fw.weight)
        else:
            w = u"0".rjust(6)
        if fw.std is None:
            yield u'{w}  {feature}'.format(feature=featname, w=w)
        else:
            yield u'{w} {plus} {std:0.4f}  {feature}'.format(
                feature=featname,
                w=w,
                plus=_PLUS_MINUS,
                std=2 * fw.std,
            )


def _decision_tree_lines(explanation):
    # type: (Explanation) -> List[str]
    assert explanation.decision_tree is not None
    return ["", tree2text(explanation.decision_tree)]


def _transition_features_lines(explanation):
    # type: (Explanation) -> List[str]
    from tabulate import tabulate  # type: ignore
    tf = explanation.transition_features
    assert tf is not None
    return [
        "",
        "Transition features:",
        tabulate(tf.coef, headers=tf.class_names, showindex=tf.class_names,
                 floatfmt="0.3f"),
        ""
    ]


def _targets_lines(explanation,  # type: Explanation
                   hl_spaces,  # type: Optional[bool]
                   show_feature_values,  # type: bool
                   explaining_prediction,  # type: bool
                   ):
    # type: (...) -> List[str]
    lines = []
    assert explanation.targets is not None
    for target in explanation.targets:
        scores = _format_scores(target.proba, target.score)
        if scores:
            scores = " (%s)" % scores

        header = "%s%r%s top features" % (
            'y=' if not explanation.is_regression else '',
            target.target,
            scores)
        lines.append(header)

        if explaining_prediction:
            table_header = ['Contribution', 'Feature']
        else:
            table_header = ['Weight', 'Feature']
        if show_feature_values:
            table_header.append('Value')
            table_line = lambda fw: [
                format_weight(fw.weight),
                _format_feature(fw.feature, hl_spaces),
                format_value(fw.value)]
            col_align = 'rlr'
        else:
            table_line = lambda fw: [
                format_weight(fw.weight),
                _format_feature(fw.feature, hl_spaces)]
            col_align = 'rl'

        w = target.feature_weights
        table = tabulate(
            [table_line(fw) for fw in chain(w.pos, reversed(w.neg))],
            header=table_header,
            col_align=col_align,
        )
        max_width = len(table[1])
        pos_table = '\n'.join(table[:-len(w.neg)])
        neg_table = '\n'.join(table[-len(w.neg):])

        if pos_table:
            lines.append(pos_table)
        if w.pos_remaining:
            lines.append(
                _format_remaining(w.pos_remaining, 'positive', max_width))
        if w.neg_remaining:
            lines.append(
                _format_remaining(w.neg_remaining, 'negative', max_width))
        if neg_table:
            lines.append(neg_table)

        lines.append('')
    return lines


def _format_scores(proba, score):
    # type: (Optional[float], Optional[float]) -> str
    scores = []
    if proba is not None:
        scores.append("probability=%0.3f" % proba)
    if score is not None:
        scores.append("score=%0.3f" % score)
    return ", ".join(scores)


def _format_remaining(remaining, kind, width):
    # type: (int, str, int) -> str
    s = '{ellipsis} {remaining} more {kind}{ellipsis}'.format(
        ellipsis=_ELLIPSIS,
        remaining=remaining,
        kind=(kind + ' ') if kind else '',
    )
    return ('{:^%d}' % width).format(s)


def _format_feature(name, hl_spaces):
    # type: (...) -> str
    if isinstance(name, bytes):
        name = name.decode('utf8')
    if isinstance(name, FormattedFeatureName):
        return name.format()
    elif isinstance(name, list) and \
            all('name' in x and 'sign' in x for x in name):
        return _format_unhashed_feature(name, hl_spaces=hl_spaces)
    else:
        return _format_single_feature(name, hl_spaces=hl_spaces)


def _format_single_feature(feature, hl_spaces):
    # type: (str, bool) -> str
    if hl_spaces:
        return replace_spaces(feature, lambda n, _: _SPACE * n)
    else:
        return feature


def _format_unhashed_feature(name, hl_spaces, sep=' | '):
    # type: (List, bool, str) -> str
    """
    Format feature name for hashed features.
    """
    return sep.join(
        format_signed(n, _format_single_feature, hl_spaces=hl_spaces)
        for n in name)
