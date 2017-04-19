# -*- coding: utf-8 -*-
import numpy as np
from eli5.base import FeatureImportances
from eli5.utils import argsort_k_largest_positive


def get_feature_importances_filtered(coef, feature_names, flt_indices, top,
                                     coef_std=None):
    if flt_indices is not None:
        coef = coef[flt_indices]
        if coef_std is not None:
            coef_std = coef_std[flt_indices]

    indices = argsort_k_largest_positive(coef, top)
    names, values = feature_names[indices], coef[indices]
    std = None if coef_std is None else coef_std[indices]
    return FeatureImportances.from_names_values(
        names, values, std,
        remaining=np.count_nonzero(coef) - len(indices),
    )
