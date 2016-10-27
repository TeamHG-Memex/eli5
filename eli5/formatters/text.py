# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six

from . import fields
from .features import FormattedFeatureName
from .utils import format_signed, replace_spaces


_PLUS_MINUS = "+-" if six.PY2 else "±"
_ELLIPSIS = '...' if six.PY2 else '…'
_SPACE = '_' if six.PY2 else '░'


def format_as_text(explanation, show=fields.ALL):
    lines = []

    for key in show:
        if key not in explanation:
            continue

        if key == 'method':
            lines.append('Explained as: {}'.format(explanation['method']))

        if key == 'description':
            lines.append(explanation['description'])

        if key == 'classes':
            lines.extend(_format_weights(explanation['classes']))

        if key == 'targets':
            lines.extend(_format_weights(explanation['targets']))

        if key == 'feature_importances':
            sz = _maxlen(explanation['feature_importances'])
            for name, w, std in explanation['feature_importances']:
                lines.append('{w:0.4f} {plus} {std:0.4f} {feature}'.format(
                    feature=name.ljust(sz),
                    w=w,
                    plus=_PLUS_MINUS,
                    std=2*std,
                ))

    return '\n'.join(lines)


def _format_weights(explanations):
    lines = []
    sz = _max_feature_size(explanations)
    for explanation in explanations:
        scores = _format_scores(
            explanation.get('proba'),
            explanation.get('score'),
        )
        if scores:
            scores = " (%s)" % scores

        if 'class' in explanation:
            header = "y=%r%s top features" % (
                explanation['class'],
                scores
            )
        elif 'target' in explanation:
            header = "%r%s top features" % (
                explanation['target'],
                scores
            )
        else:
            raise ValueError('Expected "class" or "target" key')
        lines.append(header)
        lines.append("-" * (sz + 10))

        w = explanation['feature_weights']
        lines.extend(_format_feature_weights(w['pos'], sz))
        if w['pos_remaining']:
            lines.append(_format_remaining(w['pos_remaining'], 'positive'))
        if w['neg_remaining']:
            lines.append(_format_remaining(w['neg_remaining'], 'negative'))
        lines.extend(_format_feature_weights(w['neg'], sz))
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
    return max(len(_format_feature(it[0])) for it in feature_weights)


def _max_feature_size(explanation):
    def _max_feature_length(w):
        return _maxlen(w['pos'] + w['neg'])
    return max(_max_feature_length(e['feature_weights']) for e in explanation)


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
        return str(name)
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
