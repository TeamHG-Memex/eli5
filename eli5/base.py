# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union

from .base_utils import attrs
from .formatters.features import FormattedFeatureName


# @attrs decorator used in this file calls @attr.s(slots=True),
# creating attr.ib entries based on the signature of __init__.


@attrs
class Explanation(object):
    """ An explanation for classifier or regressor,
    it can either explain weights or a single prediction.
    """
    def __init__(self,
                 estimator,  # type: str
                 description=None,  # type: str
                 error=None,  # type: str
                 method=None,  # type: str
                 is_regression=False,  # type: bool
                 targets=None,  # type: List[TargetExplanation]
                 feature_importances=None,  # type: List[FeatureWeight]
                 decision_tree=None,  # type: TreeInfo
                 highlight_spaces=None,
                 transition_features=None,  # type: TransitionFeatureWeights
                 ):
        self.estimator = estimator
        self.description = description
        self.error = error
        self.method = method
        self.is_regression = is_regression
        self.targets = targets
        self.feature_importances = feature_importances
        self.decision_tree = decision_tree
        self.highlight_spaces = highlight_spaces
        self.transition_features = transition_features

    def _repr_html_(self):
        """ HTML formatting for the notebook.
        """
        from eli5.formatters import fields
        from eli5.formatters.html import format_as_html
        return format_as_html(self, force_weights=False, show=fields.WEIGHTS)


@attrs
class TargetExplanation(object):
    """ Explanation for a single target or class.
    Feature weights are stored in the :feature_weights: attribute,
    and features highlighted in text in the :weighted_spans: attribute.
    """
    def __init__(self,
                 target,  # type: str
                 feature_weights,  # type: FeatureWeights
                 proba=None,  # type: float
                 score=None,  # type: float
                 weighted_spans=None,  # type: List[WeightedSpans]
                 ):
        self.target = target
        self.feature_weights = feature_weights
        self.proba = proba
        self.score = score
        self.weighted_spans = weighted_spans


# List is currently used for unhashed features
Feature = Union[str, List, FormattedFeatureName]


@attrs
class FeatureWeights(object):
    """ Weights for top features, :pos: for positive and :neg: for negative,
    sorted by descending absolute value.
    Number of remaining positive and negative features are stored in
    :pos_remaining: and :neg_remaining: attributes.
    """
    def __init__(self,
                 pos,  # type: List[FeatureWeight]
                 neg,  # type: List[FeatureWeight]
                 pos_remaining=0,  # type: int
                 neg_remaining=0,  # type: int
                 ):
        self.pos = pos
        self.neg = neg
        self.pos_remaining = pos_remaining
        self.neg_remaining = neg_remaining


@attrs
class FeatureWeight(object):
    def __init__(self,
                 feature,  # type: Feature
                 weight,  # type: float
                 std=None,  # type: float
                 ):
        self.feature = feature
        self.weight = weight
        self.std = std


WeightedSpan = Tuple[
    Feature,
    List[Tuple[int, int]],  # list of spans (start, end) for this feature
    float,  # feature weight
]


@attrs
class WeightedSpans(object):
    """ Features highlighted in text. :analyzer: is a type of the analyzer
    (for example "char" or "word"), and :document: is a pre-processed document
    before applying the analyzed. :weighted_spans: holds a list of spans
    (see above) for features found in text (span indices correspond to
    :document:), and :other: holds weights for features not highlighted in text.
    """
    def __init__(self,
                 analyzer,  # type: str
                 document,  # type: str
                 weighted_spans,  # type: List[WeightedSpan]
                 other=None,  # type: FeatureWeights
                 vec_name=None,  # type: str
                 ):
        self.analyzer = analyzer
        self.document = document
        self.weighted_spans = weighted_spans
        self.other = other
        self.vec_name = vec_name


@attrs
class TransitionFeatureWeights(object):
    """ Weights matrix for transition features. """
    def __init__(self,
                 class_names,  # type: List[str],
                 coef,
                 ):
        self.class_names = class_names
        self.coef = coef


@attrs
class TreeInfo(object):
    """ Information about the decision tree. :criterion: is the name of
    the function to measure the quality of a split, :tree: holds all nodes
    of the tree, and :graphviz: is the tree rendered in graphviz .dot format.
    """
    def __init__(self,
                 criterion,  # type: str
                 tree,  # type: NodeInfo
                 graphviz,  # type: str
                 ):
        self.criterion = criterion
        self.tree = tree
        self.graphviz = graphviz


@attrs
class NodeInfo(object):
    """ A node in a binary tree.
    Pointers to left and right children are in :left: and :right: attributes.
    """
    def __init__(self,
                 id,
                 is_leaf,  # type: bool
                 value,
                 value_ratio,
                 impurity,
                 samples,
                 sample_ratio,
                 feature_name=None,
                 feature_id=None,
                 threshold=None,
                 left=None,  # type: NodeInfo
                 right=None,  # type: NodeInfo
                 ):
        self.id = id
        self.is_leaf = is_leaf
        self.value = value
        self.value_ratio = value_ratio
        self.impurity = impurity
        self.samples = samples
        self.sample_ratio = sample_ratio
        self.feature_name = feature_name
        self.feature_id = feature_id
        self.threshold = threshold
        self.left = left
        self.right = right
