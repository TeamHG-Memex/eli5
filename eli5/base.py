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
    pos_remaining = attr.ib(default=0)
    neg_remaining = attr.ib(default=0)


@attr.s
class WeightedSpans(object):
    analyzer = attr.ib()
    document = attr.ib()
    weighted_spans = attr.ib()
    other = attr.ib(default=None)


@attr.s
class TreeInfo(object):
    criterion = attr.ib()
    tree = attr.ib()
    graphviz = attr.ib()


@attr.s
class NodeInfo(object):
    id = attr.ib()
    is_leaf = attr.ib()
    value = attr.ib()
    value_ratio = attr.ib()
    impurity = attr.ib()
    samples = attr.ib()
    sample_ratio = attr.ib()
    feature_name = attr.ib(default=None)
    # for non-leafs
    feature_id = attr.ib(default=None)
    threshold = attr.ib(default=None)
    left = attr.ib(default=None)
    right = attr.ib(default=None)
