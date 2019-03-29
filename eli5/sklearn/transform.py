"""transform_feature_names implementations for scikit-learn transformers
"""

import numpy as np  # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion  # type: ignore
from sklearn.feature_selection.base import SelectorMixin  # type: ignore

from sklearn.preprocessing import (  # type: ignore
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)

from eli5.transform import transform_feature_names
from eli5.sklearn.utils import get_feature_names as _get_feature_names


# Feature selection:

@transform_feature_names.register(SelectorMixin)
def _select_names(est, in_names=None):
    mask = est.get_support(indices=False)
    in_names = _get_feature_names(est, feature_names=in_names,
                                  num_features=len(mask))
    return [in_names[i] for i in np.flatnonzero(mask)]

try:
    from sklearn.linear_model import (  # type: ignore
        RandomizedLogisticRegression,
        RandomizedLasso,
    )
    _select_names = transform_feature_names.register(RandomizedLasso)(_select_names)
    _select_names = transform_feature_names.register(RandomizedLogisticRegression)(_select_names)
except ImportError:     # Removed in scikit-learn 0.21
    pass


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
