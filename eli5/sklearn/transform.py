"""transform_feature_names implementations for scikit-learn transformers
"""

import numpy as np  # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion  # type: ignore
from sklearn.feature_selection.base import SelectorMixin  # type: ignore
from sklearn.linear_model import (  # type: ignore
    RandomizedLogisticRegression,
    RandomizedLasso,
)

from eli5.transform import transform_feature_names
from eli5.sklearn.utils import get_feature_names as _get_feature_names


# Feature selection:

@transform_feature_names.register(SelectorMixin)
@transform_feature_names.register(RandomizedLogisticRegression)
@transform_feature_names.register(RandomizedLasso)
def _select_names(est, in_names=None):
    mask = est.get_support(indices=False)
    in_names = _get_feature_names(est, feature_names=in_names,
                                  num_features=len(mask))
    return [in_names[i] for i in np.flatnonzero(mask)]


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
