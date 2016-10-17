# -*- coding: utf-8 -*-
from __future__ import absolute_import
import six

_PLUS_MINUS = "+-" if six.PY2 else "±"


def format_as_text(explanation):
    lines = []
    if 'method' in explanation:
        lines.append("Explained as: %s" % explanation['method'])

    if 'description' in explanation:
        lines.append(explanation['description'])

    if 'classes' in explanation:
        lines.extend(_format_weights(explanation['classes']))

    if 'targets' in explanation:
        lines.extend(_format_weights(explanation['targets']))

    if 'feature_importances' in explanation:
        sz = _maxlen(explanation['feature_importances'])
        for name, w, std in explanation['feature_importances']:
            lines.append("%s %0.4f %s %0.4f" % (
                name.rjust(sz), w, _PLUS_MINUS, 2*std,
            ))

    return "\n".join(lines)


def _format_weights(explanations):
    lines = []
    sz = _rjust_size(explanations)
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
            msg = "%s   (%d more positive features)" % (
                "...".rjust(sz), w['pos_remaining']
            )
            lines.append(msg)
        if w['neg_remaining']:
            msg = "%s   (%d more negative features)" % (
                "...".rjust(sz), w['neg_remaining']
            )
            lines.append(msg)
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


def _rjust_size(explanation):
    def _max_feature_length(w):
        return _maxlen(w['pos'] + w['neg'])
    return max(_max_feature_length(e['feature_weights']) for e in explanation)


def _format_feature_weights(feature_weights, sz):
    return ['%s %+8.3f' % (_format_feature(name).rjust(sz), coef)
            for name, coef in feature_weights]


def _format_feature(name):
    if isinstance(name, list) and ('name' in x and 'sign' in x for x in name):
        return _format_unhashed_feature(name)
    else:
        return name


def _format_unhashed_feature(name, sep=' | '):
    """
    Format feature name for hashed features.
    """
    return sep.join(map(format_signed, name))


def format_signed(feature):
    """
    Format unhashed feature with sign.

    >>> format_signed([{'name': 'foo', 'sign': 1}])
    'foo'
    >>> format_signed([{'name': 'foo', 'sign': -1}])
    '(-)foo'
    """
    txt = '' if feature['sign'] > 0 else '(-)'
    return ''.join([txt, feature['name']])
