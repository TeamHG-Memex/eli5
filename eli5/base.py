# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union

import attr


@attr.s
class Explanation(object):
    estimator = attr.ib()  # type: str
    description = attr.ib(default=None)  # type: str
    method = attr.ib(default=None)  # type: str
    targets = attr.ib(default=None)  # type: List[TargetExplanation]
    is_regression = attr.ib(default=False)  # type: bool
    feature_importances = attr.ib(default=None)  # type: FeatureWeights
    decision_tree = attr.ib(default=None)  # type: TreeInfo

    def _repr_html_(self):
        from eli5.formatters import format_as_html, fields
        if self.targets or self.feature_importances or self.decision_tree:
            show = fields.WEIGHTS
        else:
            show = fields.ALL
        return format_as_html(self, force_weights=False, show=show)


@attr.s
class TargetExplanation(object):
    target = attr.ib()  # type: str
    feature_weights = attr.ib()  # type: FeatureWeights
    proba = attr.ib(default=None)  # type: float
    score = attr.ib(default=None)  # type: float
    weighted_spans = attr.ib(default=None)  # type: WeightedSpans


Feature = Union[str, Dict]

@attr.s
class FeatureWeights(object):
    pos = attr.ib()  # type: List[Tuple[Feature, float]]
    neg = attr.ib()  # type: List[Tuple[Feature, float]]
    pos_remaining = attr.ib(default=0)  # type: int
    neg_remaining = attr.ib(default=0)  # type: int


WeightedSpan = Tuple[Feature, List[Tuple[int, int]], float]

@attr.s
class WeightedSpans(object):
    analyzer = attr.ib()  # type: str
    document = attr.ib()  # type: str
    weighted_spans = attr.ib()  # type: List[WeightedSpan]
    other = attr.ib(default=None)  # type: FeatureWeights


@attr.s
class TreeInfo(object):
    criterion = attr.ib()  # type: str
    tree = attr.ib()  # type: NodeInfo
    graphviz = attr.ib()  # type: str


@attr.s
class NodeInfo(object):
    id = attr.ib()
    is_leaf = attr.ib()  # type: bool
    value = attr.ib()
    value_ratio = attr.ib()
    impurity = attr.ib()
    samples = attr.ib()
    sample_ratio = attr.ib()
    feature_name = attr.ib(default=None)
    # for non-leafs
    feature_id = attr.ib(default=None)
    threshold = attr.ib(default=None)
    left = attr.ib(default=None)  # type: NodeInfo
    right = attr.ib(default=None)  # type: NodeInfo
