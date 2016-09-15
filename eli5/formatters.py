# -*- coding: utf-8 -*-
"""
Functions to convert explanations to human-digestable formats.

TODO: html, IPython integration, customizability.
"""
from __future__ import absolute_import
import six

_PLUS_MINUS = "+-" if six.PY2 else "±"


def format_as_text(explanation):
    lines = []
    lines.append(explanation['classifier'])
    if 'method' in explanation:
        lines.append("Explained as: %s" % explanation['method'])

    if 'description' in explanation:
        lines.append(explanation['description'])

    if 'classes' in explanation:
        sz = _rjust_size(explanation['classes'])
        for class_explanation in explanation['classes']:
            lines.append("y=%r top features" % class_explanation['class'])
            lines.append("-" * (sz + 10))
            weights = class_explanation['feature_weights']
            pos = weights['pos']
            neg = weights['neg']
            for name, w in pos:
                lines.append("%s %+8.3f" % (name.rjust(sz), w))
            if weights['truncated']:
                lines.append("...".rjust(sz))
            for name, w in reversed(neg):
                lines.append("%s %8.3f" % (name.rjust(sz), w))
            lines.append("")

    if 'feature_importances' in explanation:
        sz = _maxlen(explanation['feature_importances'])
        for name, w, std in explanation['feature_importances']:
            lines.append("%s %0.4f %s %0.4f" % (
                name.rjust(sz), w, _PLUS_MINUS, 2*std,
            ))

    return "\n".join(lines)


def _maxlen(features):
    if not features:
        return 0
    return max(len(it[0]) for it in features)


def _rjust_size(explanation):
    def _max_feature_length(w):
        return _maxlen(w['pos'] + w['neg'])
    return max(_max_feature_length(e['feature_weights']) for e in explanation)
