# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .explain_weights import (
    explain_weights_sklearn,
    explain_linear_classifier_weights,
    explain_linear_regressor_weights,
    explain_rf_feature_importance,
    explain_decision_tree,
)
from .explain_prediction import (
    explain_prediction_sklearn,
    explain_prediction_linear_classifier,
    explain_prediction_linear_regressor,
)
from .unhashing import (
    InvertableHashingVectorizer,
    FeatureUnhasher,
    invert_hashing_and_fit,
)
from .permutation_importance import PermutationImportance
from . import transform as _
