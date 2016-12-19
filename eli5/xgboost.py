# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re
from singledispatch import singledispatch
from typing import List, Dict

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from eli5.base import FeatureWeight, FeatureImportances, Explanation
from eli5.explain import explain_weights
from eli5.sklearn.utils import get_feature_names
from eli5.utils import argsort_k_largest_positive


DESCRIPTION_XGBOOST = """
XGBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


@explain_weights.register(XGBClassifier)
@explain_weights.register(XGBRegressor)
@singledispatch
def explain_weights_xgboost(xgb,
                            vec=None,
                            top=20,
                            target_names=None,  # ignored
                            targets=None,  # ignored
                            feature_names=None,
                            feature_re=None):
    """
    Return an explanation of an XGBoost estimator (via scikit-learn wrapper).
    """
    feature_names = get_feature_names(xgb, vec, feature_names=feature_names)
    coef = xgb.feature_importances_

    if feature_re is not None:
        feature_names, flt_indices = feature_names.filtered_by_re(feature_re)
        coef = coef[flt_indices]

    indices = argsort_k_largest_positive(coef, top)
    names, values = feature_names[indices], coef[indices]
    return Explanation(
        feature_importances=FeatureImportances(
            [FeatureWeight(*x) for x in zip(names, values)],
            remaining=np.count_nonzero(coef) - len(indices),
        ),
        description=DESCRIPTION_XGBOOST,
        estimator=repr(xgb),
        method='feature importances',
    )


def parse_dump(text_dump):
    result = None
    stack = []  # type: List[Dict]
    for line in text_dump.split('\n'):
        if line:
            depth, node = _parse_dump_line(line)
            if depth == 0:
                assert not stack
                result = node
                stack.append(node)
            elif depth > len(stack):
                raise ValueError('Unexpected dump structure')
            else:
                if depth < len(stack):
                    stack = stack[:depth]
                stack[-1].setdefault('children', []).append(node)
                stack.append(node)
    return result


def _parse_dump_line(line):
    branch_match = re.match(
        '^(\t*)(\d+):\[(\w+)<([^\]]+)\] yes=(\d+),no=(\d+),missing=(\d+)$', line)
    if branch_match:
        tabs, node_id, feature, condition, yes, no, missing = \
            branch_match.groups()
        depth = len(tabs)
        return depth, {
            'depth': depth,
            'nodeid': int(node_id),
            'split': feature,
            'split_condition': float(condition),
            'yes': int(yes),
            'no': int(no),
            'missing': int(missing),
        }
    leaf_match = re.match('^(\t*)(\d+):leaf=(.*)$', line)
    if leaf_match:
        tabs, node_id, value = leaf_match.groups()
        depth = len(tabs)
        return depth, {
            'nodeid': int(node_id),
            'leaf': float(value),
        }
    raise ValueError('Line in unexpected format: {}'.format(line))
