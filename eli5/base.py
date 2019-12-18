# -*- coding: utf-8 -*-
from typing import Any, List, Tuple, Union, Optional

import numpy as np

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
                 description=None,  # type: Optional[str]
                 error=None,  # type: Optional[str]
                 method=None,  # type: Optional[str]
                 is_regression=False,  # type: bool
                 targets=None,  # type: Optional[List[TargetExplanation]]
                 feature_importances=None,  # type: Optional[FeatureImportances]
                 decision_tree=None,  # type: Optional[TreeInfo]
                 highlight_spaces=None,  # type: Optional[bool]
                 transition_features=None,  # type: Optional[TransitionFeatureWeights]
                 image=None, # type: Any
                 ):
        # type: (...) -> None
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
        self.image = image # if arg is not None, assume we are working with images

    def _repr_html_(self):
        """ HTML formatting for the notebook.
        """
        from eli5.formatters import fields
        from eli5.formatters.html import format_as_html
        return format_as_html(self, force_weights=False, show=fields.WEIGHTS)


@attrs
class FeatureImportances(object):
    """ Feature importances with number of remaining non-zero features.
    """
    def __init__(self, importances, remaining):
        # type: (...) -> None
        self.importances = importances  # type: List[FeatureWeight]
        self.remaining = remaining  # type: int

    @classmethod
    def from_names_values(cls, names, values, std=None, **kwargs):
        params = zip(names, values) if std is None else zip(names, values, std)
        importances = [FeatureWeight(*x) for x in params]  # type: ignore
        return cls(importances, **kwargs)


@attrs
class TargetExplanation(object):
    """ Explanation for a single target or class.
    Feature weights are stored in the :feature_weights: attribute,
    and features highlighted in text in the :weighted_spans: attribute.

    Spatial values are stored in the :heatmap: attribute.
    """
    def __init__(self,
                 target,  # type: Union[str, int]
                 feature_weights=None,  # type: Optional[FeatureWeights]
                 proba=None,  # type: Optional[float]
                 score=None,  # type: Optional[float]
                 weighted_spans=None,  # type: Optional[WeightedSpans]
                 heatmap=None, # type: Optional[np.ndarray]
                 ):
        # type: (...) -> None
        self.target = target
        self.feature_weights = feature_weights
        self.proba = proba
        self.score = score
        self.weighted_spans = weighted_spans
        self.heatmap = heatmap


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
        # type: (...) -> None
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
                 value=None,  # type: Any
                 ):
        # type: (...) -> None
        self.feature = feature
        self.weight = weight
        self.std = std
        self.value = value


@attrs
class WeightedSpans(object):
    """ Holds highlighted spans for parts of document - a DocWeightedSpans
    object for each vectorizer, and other features not highlighted anywhere.
    """
    def __init__(self,
                 docs_weighted_spans,  # type: List[DocWeightedSpans]
                 other=None,  # type: FeatureWeights
                 ):
        # type: (...) -> None
        self.docs_weighted_spans = docs_weighted_spans
        self.other = other


WeightedSpan = Tuple[
    Feature,
    List[Tuple[int, int]],  # list of spans (start, end) for this feature
    float,  # feature weight
]


@attrs
class DocWeightedSpans(object):
    """ Features highlighted in text. :document: is a pre-processed document
    before applying the analyzer. :weighted_spans: holds a list of spans
    for features found in text (span indices correspond to
    :document:). :preserve_density: determines how features are colored
    when doing formatting - it is better set to True for char features
    and to False for word features.
    """
    def __init__(self,
                 document,  # type: str
                 spans,  # type: List[WeightedSpan]
                 preserve_density=None,  # type: bool
                 vec_name=None,  # type: str
                 ):
        # type: (...) -> None
        self.document = document
        self.spans = spans
        self.preserve_density = preserve_density
        self.vec_name = vec_name


@attrs
class TransitionFeatureWeights(object):
    """ Weights matrix for transition features. """
    def __init__(self,
                 class_names,  # type: List[str]
                 coef,
                 ):
        # type: (...) -> None
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
                 is_classification, # type: bool
                 ):
        # type: (...) -> None
        self.criterion = criterion
        self.tree = tree
        self.graphviz = graphviz
        self.is_classification = is_classification


@attrs
class NodeInfo(object):
    """ A node in a binary tree.
    Pointers to left and right children are in :left: and :right: attributes.
    """
    def __init__(self,
                 id,                 # type: int
                 is_leaf,            # type: bool
                 value,
                 value_ratio,
                 impurity,           # type: float
                 samples,            # type: int
                 sample_ratio,       # type: float
                 feature_name=None,  # type: str
                 feature_id=None,    # type: int
                 threshold=None,     # type: float
                 left=None,          # type: NodeInfo
                 right=None,         # type: NodeInfo
                 ):
        # type: (...) -> None
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
