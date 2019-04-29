"""transform_feature_names implementations for scikit-learn transformers
"""

import numpy as np  # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion  # type: ignore
from sklearn.feature_selection.base import SelectorMixin  # type: ignore
try:
    from sklearn.linear_model import (  # type: ignore
        RandomizedLogisticRegression,
        RandomizedLasso,
    )
except ImportError:     
    # randomized_l1 feature selectors are not available (removed in scikit-learn 0.21)
    RandomizedLogisticRegression = None
    RandomizedLasso = None
try:
    from stability_selection import StabilitySelection
    # TODO: add support for stability_selection.RandomizedLogisticRegression and stability_selection.RandomizedLasso ?
except ImportError:
    # scikit-learn-contrib/stability-selection is not available
    StabilitySelection = None

from sklearn.preprocessing import (  # type: ignore
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)

from eli5.transform import transform_feature_names
from eli5.sklearn.utils import get_feature_names as _get_feature_names


def register_notnone(generic_func, cls):
    """
    Register an implementation of a generic function 
    if the supplied type is not None.
    """
    def inner_register(func):
        if cls is None:
            # do nothing
            return func
        else:
            # register a new implementation
            return generic_func.register(cls)(func)
    return inner_register


# Feature selection:

@transform_feature_names.register(SelectorMixin)
@register_notnone(transform_feature_names, RandomizedLogisticRegression)
@register_notnone(transform_feature_names, RandomizedLasso)
@register_notnone(transform_feature_names, StabilitySelection)
def _select_names(est, in_names=None):
    mask = est.get_support(indices=False)
    in_names = _get_feature_names(est, feature_names=in_names,
                                  num_features=len(mask))
    return [in_names[i] for i in np.flatnonzero(mask)]


# Scaling

@transform_feature_names.register(MinMaxScaler)
@transform_feature_names.register(StandardScaler)
@transform_feature_names.register(MaxAbsScaler)
@transform_feature_names.register(RobustScaler)
def _transform_scaling(est, in_names=None):
    if in_names is None:
        in_names = _get_feature_names(est, feature_names=in_names,
                                      num_features=est.scale_.shape[0])
    return [name for name in in_names]


# Pipelines

@transform_feature_names.register(Pipeline)
def _pipeline_names(est, in_names=None):
    names = in_names
    for name, trans in est.steps:
        if trans is not None:
            names = transform_feature_names(trans, names)
    return names


@transform_feature_names.register(FeatureUnion)
def _union_names(est, in_names=None):
    return ['{}:{}'.format(trans_name, feat_name)
            for trans_name, trans, _ in est._iter()
            for feat_name in transform_feature_names(trans, in_names)]
