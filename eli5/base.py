# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union

import attr


@attr.s
class Explanation(object):
    """ An explanation for classifier or regressor,
    it can either explain weights or a single prediction.
    """
    # Explanation meta-information
    estimator = attr.ib()  # type: str
    description = attr.ib(default=None)  # type: str
    method = attr.ib(default=None)  # type: str
    is_regression = attr.ib(default=False)  # type: bool
    # Actual explanations
    targets = attr.ib(default=None)  # type: List[TargetExplanation]
    feature_importances = attr.ib(default=None)  # type: FeatureWeights
    decision_tree = attr.ib(default=None)  # type: TreeInfo

    def _repr_html_(self):
        from eli5.formatters import format_as_html, fields
        return format_as_html(self, force_weights=False, show=fields.WEIGHTS)


@attr.s
class TargetExplanation(object):
    """ Explanation for a single target or class.
    Feature weights are stored in the :feature_weights: attribute,
    and features highlighted in text in the :weighted_spans: attribute.
    """
    target = attr.ib()  # type: str
    feature_weights = attr.ib()  # type: FeatureWeights
    proba = attr.ib(default=None)  # type: float
    score = attr.ib(default=None)  # type: float
    weighted_spans = attr.ib(default=None)  # type: WeightedSpans


Feature = Union[str, Dict]  # Dict is currently used for unhashed features


@attr.s
class FeatureWeights(object):
    """ Weights for top features, :pos: for positive and :neg: for negative,
    sorted by descending absolute value.
    Number of remaining positive and negative features are store in
    :pos_remaining: and :neg_remaining: attributes.
    """
    pos = attr.ib()  # type: List[Tuple[Feature, float]]
    neg = attr.ib()  # type: List[Tuple[Feature, float]]
    pos_remaining = attr.ib(default=0)  # type: int
    neg_remaining = attr.ib(default=0)  # type: int


WeightedSpan = Tuple[
    Feature,
    List[Tuple[int, int]],  # list of spans (start, end) for this feature
    float,  # feature weight
]


@attr.s
class WeightedSpans(object):
    """ Features highlighted in text. :analyzer: is a type of the analyzer
    (for example "char" or "word"), and :document: is a pre-processed document
    before applying the analyzed. :weighted_spans: holds a list of spans
    (see above) for features found in text (span indices correspond to :document:),
    and :other: holds weights for features not highlighted in text.
    """
    analyzer = attr.ib()  # type: str
    document = attr.ib()  # type: str
    weighted_spans = attr.ib()  # type: List[WeightedSpan]
    other = attr.ib(default=None)  # type: FeatureWeights


@attr.s
class TreeInfo(object):
    """ Information about the decision tree. :criterion: is the name of the function
    to measure the quality of a split, :tree: holds all nodes of the tree, and
    :graphviz: is the tree rendered in graphviz .dot format.
    """
    criterion = attr.ib()  # type: str
    tree = attr.ib()  # type: NodeInfo
    graphviz = attr.ib()  # type: str


@attr.s
class NodeInfo(object):
    """ A node in a binary tree.
    Pointers to left and right children are in :left: and :right: attributes.
    """
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
