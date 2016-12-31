# -*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import defaultdict
import re
from singledispatch import singledispatch
import six
from typing import Any, Dict, List, Tuple, Union

import numpy as np  # type: ignore
import scipy.sparse as sp  # type: ignore
from xgboost import (  # type: ignore
    XGBClassifier,
    XGBRegressor,
    Booster,
    DMatrix
)

from eli5.base import (
    FeatureWeight, FeatureImportances, Explanation, TargetExplanation)
from eli5.formatters.features import FormattedFeatureName
from eli5.explain import explain_weights, explain_prediction
from eli5.sklearn.text import add_weighted_spans
from eli5.sklearn.utils import (
    get_feature_names, get_X, handle_vec, predict_proba)
from eli5.utils import argsort_k_largest_positive, get_target_display_names
from eli5._decision_path import DECISION_PATHS_CAVEATS
from eli5._feature_weights import get_top_features


DECISION_PATHS_CAVEATS = """
Feature weights are calculated by following decision paths in trees
of an ensemble. Each leaf has an output score, and expected scores can also be
assigned to parent nodes. Contribution of one feature on the decision path
is how much expected score changes from parent to child. Weights of all features
sum to the output score of the estimator.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_XGBOOST = """
XGBoost feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

DESCRIPTION_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_CLF_BINARY = """
Features with largest coefficients.
""" + DECISION_PATHS_CAVEATS

DESCRIPTION_REGRESSION = DESCRIPTION_CLF_BINARY


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
    Return an explanation of an XGBoost estimator (via scikit-learn wrapper
    XGBClassifier or XGBRegressor) as feature importances.
    """
    coef = _xgb_feature_importances(xgb)
    num_features = coef.shape[-1]
    feature_names = get_feature_names(
        xgb, vec, feature_names=feature_names, num_features=num_features)

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
        is_regression=isinstance(xgb, XGBRegressor),
    )


@explain_prediction.register(XGBClassifier)
@explain_prediction.register(XGBRegressor)
@singledispatch
def explain_prediction_xgboost(
        xgb, doc,
        vec=None,
        top=None,
        target_names=None,
        targets=None,
        feature_names=None,
        vectorized=False,
        expand_missing_features=False,
    ):
    """ Return an explanation of XGBoost prediction (via scikit-learn wrapper
    XGBClassifier or XGBRegressor) as feature weights.

    Method for determining feature importances follows an idea from
    http://blog.datadive.net/interpreting-random-forests/.
    Feature weights are calculated by following decision paths in trees
    of an ensemble.
    Each leaf has an output score, and expected scores can also be assigned
    to parent nodes.
    Contribution of one feature on the decision path is how much expected score
    changes from parent to child.
    Weights of all features sum to the output score of the estimator.

    For correct handling of missing features, set missing argument of
    XGBClassifier or XGBRegressor correctly (e.g. to 0 when working with
    sklearn text vectorizers).
    When expand_missing_features is set to True, each missing feature is
    reported separately, as "feature (missing)"; when it is False,
    all missing features weights are added together.
    """
    num_features = len(xgb.booster().feature_names)
    vec, feature_names = handle_vec(
        xgb, doc, vec, vectorized, feature_names, num_features=num_features)
    if feature_names.bias_name is None:
        # XGBoost estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    X = get_X(doc, vec, vectorized=vectorized)
    if sp.issparse(X):
        # Work around XGBoost issue:
        # https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        X = X.tocsc()

    proba = predict_proba(xgb, X)
    scores_weights = _prediction_feature_weights(xgb, X, feature_names)

    is_multiclass = _xgb_n_targets(xgb) > 1
    is_regression = isinstance(xgb, XGBRegressor)
    names = xgb.classes_ if not is_regression else ['y']
    display_names = get_target_display_names(names, target_names, targets)

    # map feature idx to missing feature idx
    missing_map = {}  # type: Dict[Union[int, str], int]
    for _, (feature_weights, missing_feature_weights) in scores_weights:
        if expand_missing_features:
            for idx, value in six.iteritems(missing_feature_weights):
                if idx not in missing_map:
                    missing_map[idx] = feature_names.add_feature(
                        '{} (missing)'.format(feature_names[idx]))
                feature_weights[missing_map[idx]] += value
        elif missing_feature_weights:
            if 'missing' not in missing_map:
                missing = FormattedFeatureName('Missing features')
                missing_map['missing'] = feature_names.add_feature(missing)
            feature_weights[missing_map['missing']] = sum(
                missing_feature_weights.values())

    def get_weights(label_id):
        score, (feature_weights, missing_feature_weights) = (
            scores_weights[label_id])
        feature_weights_array = np.zeros(len(feature_names))
        for idx, value in six.iteritems(feature_weights):
            # Couldn't have used bias_idx before,
            # because feature_names were changing, and so was bias_idx
            idx = feature_names.bias_idx if idx == 'bias' else idx
            feature_weights_array[idx] = value
        return score, get_top_features(
            feature_names, feature_weights_array, top)

    res = Explanation(
        estimator=repr(xgb),
        method='decision paths',
        description={
            (False, False): DESCRIPTION_CLF_BINARY,
            (False, True): DESCRIPTION_CLF_MULTICLASS,
            (True, False): DESCRIPTION_REGRESSION,
        }[is_regression, is_multiclass],
        is_regression=is_regression,
        targets=[],
    )
    if is_multiclass:
        for label_id, label in display_names:
            score, feature_weights = get_weights(label_id)
            target_expl = TargetExplanation(
                target=label,
                feature_weights=feature_weights,
                score=score,
                proba=proba[label_id] if proba is not None else None,
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            res.targets.append(target_expl)
    else:
        score, feature_weights = get_weights(0)
        target_expl = TargetExplanation(
            target=display_names[-1][1],
            feature_weights=feature_weights,
            score=score,
            proba=proba[1] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


def _prediction_feature_weights(xgb, X, feature_names):
    """ For each target, return score and a tuple of feature weights
    and missing feature weights on this prediction,
    following an idea from
    http://blog.datadive.net/interpreting-random-forests/
    """
    # XGBClassifier does not have pred_leaf argument, so use booster
    booster = xgb.booster()  # type: Booster
    leaf_ids, = booster.predict(DMatrix(X, missing=xgb.missing), pred_leaf=True)
    tree_dumps = booster.get_dump(with_stats=True)
    assert len(tree_dumps) == len(leaf_ids)

    def target_feature_weights(_leaf_ids, _tree_dumps):
        return _target_feature_weights(_leaf_ids, _tree_dumps, X, xgb.missing)

    n_targets = _xgb_n_targets(xgb)
    if n_targets > 1:
        # For multiclass, XGBoost stores dumps and leaf_ids in a 1d array,
        # so we need to split them.
        scores_weights = [
            target_feature_weights(
                leaf_ids[target_idx::n_targets],
                tree_dumps[target_idx::n_targets],
            ) for target_idx in range(n_targets)]
    else:
        scores_weights = [target_feature_weights(leaf_ids, tree_dumps)]
    return scores_weights


def _target_feature_weights(leaf_ids, tree_dumps, X, missing):
    feature_weights = defaultdict(float)  # type: Dict[Union[int, str], float]
    # Union below is a lie (only ints are keys), but mypy can't ignore it
    missing_feature_weights = defaultdict(float)  # type: Dict[Union[int, str], float]
    # All trees in XGBoost give equal contribution to the prediction:
    # it is equal to sum of "leaf" values in leafs
    # before applying loss-specific function
    # (e.g. logistic for "binary:logistic" loss).
    score = 0
    for text_dump, leaf_id in zip(tree_dumps, leaf_ids):
        leaf = _indexed_leafs(_parse_tree_dump(text_dump))[leaf_id]
        score += leaf['leaf']
        path = [leaf]
        while 'parent' in path[-1]:
            path.append(path[-1]['parent'])
        path.reverse()
        # Check how each split changes "leaf" value
        for node, child in zip(path, path[1:]):
            f_num_match = re.search('^f(\d+)$', node['split'])
            feature_idx = int(f_num_match.groups()[0])
            assert feature_idx >= 0
            if _is_missing(X, feature_idx, missing):
                weights = missing_feature_weights
            else:
                weights = feature_weights
            weights[feature_idx] += child['leaf'] - node['leaf']
        # Root "leaf" value is interpreted as bias
        feature_weights['bias'] += path[0]['leaf']
    return score, (feature_weights, missing_feature_weights)


def _is_missing(X, feature_idx, missing):
    value = X[0, feature_idx]
    return np.isnan(value) if np.isnan(missing) else value == missing


def _indexed_leafs(parent):
    """ Return a leaf nodeid -> node dictionary with
    "parent" and "leaf" (average child "leaf" value) added to all nodes.
    """
    indexed = {}
    for child in parent['children']:
        child['parent'] = parent
        if 'leaf' in child:
            indexed[child['nodeid']] = child
        else:
            indexed.update(_indexed_leafs(child))
    parent['leaf'] = _parent_value(parent['children'])
    return indexed


def _parent_value(children):
    """ Value of the parent node: a weighted sum of child values.
    """
    covers = np.array([child['cover'] for child in children])
    covers /= np.sum(covers)
    leafs = np.array([child['leaf'] for child in children])
    return np.mean(leafs * covers)


def _xgb_n_targets(xgb):
    if isinstance(xgb, XGBClassifier):
        return 1 if xgb.n_classes_ == 2 else xgb.n_classes_
    elif isinstance(xgb, XGBRegressor):
        return 1
    else:
        raise TypeError


def _xgb_feature_importances(xgb):
    # XGBRegressor does not have feature_importances_ property
    # in xgboost <= 0.6a2, fixed in https://github.com/dmlc/xgboost/pull/1591
    b = xgb.booster()
    fs = b.get_fscore()
    all_features = np.array(
        [fs.get(f, 0.) for f in b.feature_names], dtype=np.float32)
    return all_features / all_features.sum()


def _parse_tree_dump(text_dump):
    """ Parse text tree dump (one item of a list returned by Booster.get_dump())
    into json format that will be used by next XGBoost release.
    """
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
    # type: (str) -> Tuple[int, Dict[str, Any]]
    branch_match = re.match(
        '^(\t*)(\d+):\[(\w+)<([^\]]+)\] '
        'yes=(\d+),no=(\d+),missing=(\d+),'
        'gain=([^,]+),cover=(.+)$', line)
    if branch_match:
        tabs, node_id, feature, condition, yes, no, missing, gain, cover = \
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
            'gain': float(gain),
            'cover': float(cover),
        }
    leaf_match = re.match('^(\t*)(\d+):leaf=([^,]+),cover=(.+)$', line)
    if leaf_match:
        tabs, node_id, value, cover = leaf_match.groups()
        depth = len(tabs)
        return depth, {
            'nodeid': int(node_id),
            'leaf': float(value),
            'cover': float(cover),
        }
    raise ValueError('Line in unexpected format: {}'.format(line))
