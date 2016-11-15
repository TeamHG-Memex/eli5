# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six
from typing import List

from . import fields
from .features import FormattedFeatureName
from .utils import format_signed, replace_spaces, should_highlight_spaces
from .trees import tree2text


_PLUS_MINUS = "+-" if six.PY2 else "±"
_ELLIPSIS = '...' if six.PY2 else '…'
_SPACE = '_' if six.PY2 else '░'


def format_as_text(expl, show=fields.ALL, highlight_spaces=None):
    """ Format explanation as text.
    If ``highlight_spaces`` is None (default), spaces will be highlighted in
    feature names only if there are any spaces at the start or at the end of the
    feature. Setting it to True forces space highlighting, and setting it to False
    turns it off.
    """
    lines = []  # type: List[str]

    if highlight_spaces is None:
        highlight_spaces = should_highlight_spaces(expl)

    if expl.error:  # always shown
        lines.extend(_error_lines(expl))

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
            lines.extend(_targets_lines(expl, hl_spaces=highlight_spaces))

        if key == 'feature_importances':
            lines.extend(_feature_importances_lines(
                expl, hl_spaces=highlight_spaces))

        if key == 'decision_tree':
            lines.extend(_decision_tree_lines(expl))

    return '\n'.join(lines)


def _method_lines(explanation):
    return ['Explained as: {}'.format(explanation.method)]


def _description_lines(explanation):
    return [explanation.description]


def _error_lines(explanation):
    return ['Error: {}'.format(explanation.error)]


def _feature_importances_lines(explanation, hl_spaces):
    sz = _maxlen(explanation.feature_importances)
    for fw in explanation.feature_importances:
        yield u'{w:0.4f} {plus} {std:0.4f} {feature}'.format(
            feature=_format_feature(fw.feature, hl_spaces).ljust(sz),
            w=fw.weight,
            plus=_PLUS_MINUS,
            std=2 * fw.std,
        )


def _decision_tree_lines(explanation):
    return ["", tree2text(explanation.decision_tree)]


def _transition_features_lines(explanation):
    from tabulate import tabulate
    tf = explanation.transition_features
    return [
        "",
        "Transition features:",
        tabulate(tf.coef, headers=tf.class_names, showindex=tf.class_names,
                 floatfmt="0.3f"),
        ""
    ]


def _targets_lines(explanation, hl_spaces):
    lines = []
    sz = _max_feature_size(explanation.targets)
    for target in explanation.targets:
        scores = _format_scores(target.proba, target.score)
        if scores:
            scores = " (%s)" % scores

        header = "%s%r%s top features" % (
            'y=' if not explanation.is_regression else '',
            target.target,
            scores)
        lines.append(header)
        lines.append("-" * (sz + 10))

        w = target.feature_weights
        lines.extend(_format_feature_weights(w.pos, sz, hl_spaces=hl_spaces))
        if w.pos_remaining:
            lines.append(_format_remaining(w.pos_remaining, 'positive'))
        if w.neg_remaining:
            lines.append(_format_remaining(w.neg_remaining, 'negative'))
        lines.extend(
            _format_feature_weights(reversed(w.neg), sz, hl_spaces=hl_spaces))
        lines.append("")
    return lines


def _format_scores(proba, score):
    scores = []
    if proba is not None:
        scores.append("probability=%0.3f" % proba)
    if score is not None:
        scores.append("score=%0.3f" % score)
    return ", ".join(scores)


def _maxlen(feature_weights):
    if not feature_weights:
        return 0
    return max(len(_format_feature(fw.feature, hl_spaces=False))
               for fw in feature_weights)


def _max_feature_size(explanation):
    def _max_feature_length(w):
        return _maxlen(w.pos + w.neg)
    return max(_max_feature_length(e.feature_weights) for e in explanation)


def _format_feature_weights(feature_weights, sz, hl_spaces):
    return [
        u'{weight:+8.3f}  {feature}'.format(
            weight=fw.weight,
            feature=_format_feature(fw.feature, hl_spaces=hl_spaces).ljust(sz))
        for fw in feature_weights]


def _format_remaining(remaining, kind):
    return '{ellipsis}  ({remaining} more {kind} features)'.format(
        ellipsis=_ELLIPSIS.rjust(8),
        remaining=remaining,
        kind=kind,
    )


def _format_feature(name, hl_spaces):
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
    if hl_spaces:
        return replace_spaces(feature, lambda n, _: _SPACE * n)
    else:
        return feature


def _format_unhashed_feature(name, hl_spaces, sep=' | '):
    """
    Format feature name for hashed features.
    """
    return sep.join(
        format_signed(n, _format_single_feature, hl_spaces=hl_spaces)
        for n in name)
