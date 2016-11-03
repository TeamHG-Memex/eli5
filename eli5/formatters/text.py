# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six

from . import fields
from .features import FormattedFeatureName
from .utils import format_signed, replace_spaces
from .trees import tree2text


_PLUS_MINUS = "+-" if six.PY2 else "±"
_ELLIPSIS = '...' if six.PY2 else '…'
_SPACE = '_' if six.PY2 else '░'


def format_as_text(expl, show=fields.ALL):
    lines = []

    if 'method' in show and expl.method:
        lines.append('Explained as: {}'.format(expl.method))

    if 'description' in show and expl.description:
        lines.append(expl.description)

    if expl.error:  # always shown
        lines.append('Error: {}'.format(expl.error))

    if 'targets' in show and expl.targets:
        lines.extend(_format_weights(expl))

    if 'feature_importances' in show and expl.feature_importances:
        sz = _maxlen(expl.feature_importances)
        for name, w, std in expl.feature_importances:
            lines.append('{w:0.4f} {plus} {std:0.4f} {feature}'.format(
                feature=name.ljust(sz),
                w=w,
                plus=_PLUS_MINUS,
                std=2*std,
            ))

    if 'decision_tree' in show and expl.decision_tree:
        lines.append("")
        lines.append(_format_decision_tree(expl.decision_tree))

    return '\n'.join(lines)


def _format_weights(explanation):
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
        lines.extend(_format_feature_weights(w.pos, sz))
        if w.pos_remaining:
            lines.append(_format_remaining(w.pos_remaining, 'positive'))
        if w.neg_remaining:
            lines.append(_format_remaining(w.neg_remaining, 'negative'))
        lines.extend(_format_feature_weights(reversed(w.neg), sz))
        lines.append("")
    return lines


def _format_scores(proba, score):
    scores = []
    if proba is not None:
        scores.append("probability=%0.3f" % proba)
    if score is not None:
        scores.append("score=%0.3f" % score)
    return ", ".join(scores)


def _format_decision_tree(treedict):
    return tree2text(treedict)


def _maxlen(feature_weights):
    if not feature_weights:
        return 0
    return max(len(_format_feature(it[0])) for it in feature_weights)


def _max_feature_size(explanation):
    def _max_feature_length(w):
        return _maxlen(w.pos + w.neg)
    return max(_max_feature_length(e.feature_weights) for e in explanation)


def _format_feature_weights(feature_weights, sz):
    return ['{weight:+8.3f}  {feature}'.format(
        weight=coef, feature=_format_feature(name).ljust(sz))
            for name, coef in feature_weights]


def _format_remaining(remaining, kind):
    return '{ellipsis}  ({remaining} more {kind} features)'.format(
        ellipsis=_ELLIPSIS.rjust(8),
        remaining=remaining,
        kind=kind,
    )


def _format_feature(name):
    if isinstance(name, FormattedFeatureName):
        return name.format()
    elif isinstance(name, list) and \
            all('name' in x and 'sign' in x for x in name):
        return _format_unhashed_feature(name)
    else:
        return _format_single_feature(name)


def _format_single_feature(feature):
    return replace_spaces(feature, lambda n, _: _SPACE * n)


def _format_unhashed_feature(name, sep=' | '):
    """
    Format feature name for hashed features.
    """
    return sep.join(format_signed(n, _format_single_feature) for n in name)
