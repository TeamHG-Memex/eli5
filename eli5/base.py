# -*- coding: utf-8 -*-

import attr


@attr.s
class Explanation(object):
    estimator = attr.ib()
    description = attr.ib(default=None)
    method = attr.ib(default=None)
    targets = attr.ib(default=None)
    is_regression = attr.ib(default=False)
    feature_importances = attr.ib(default=None)
    decision_tree = attr.ib(default=None)


# TODO - split into two classes?
@attr.s
class TargetExplanation(object):
    target = attr.ib()
    feature_weights = attr.ib()
    proba = attr.ib(default=None)
    score = attr.ib(default=None)
    weighted_spans = attr.ib(default=None)


@attr.s
class FeatureWeights(object):
    pos = attr.ib()
    neg = attr.ib()
    pos_remaining = attr.ib()
    neg_remaining = attr.ib()


@attr.s
class WeightedSpans(object):
    analyzer = attr.ib()
    document = attr.ib()
    weighted_spans = attr.ib()
    other = attr.ib()
