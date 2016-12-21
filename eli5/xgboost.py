# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re
from singledispatch import singledispatch
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from xgboost import XGBClassifier, XGBRegressor, Booster, DMatrix

from eli5.base import (
    FeatureWeight, FeatureImportances, Explanation, TargetExplanation)
from eli5.formatters.features import FormattedFeatureName
from eli5.explain import explain_weights, explain_prediction
from eli5.sklearn.text import add_weighted_spans
from eli5.sklearn.utils import (
    get_feature_names, get_X, handle_vec, predict_proba)
from eli5.utils import argsort_k_largest_positive, get_target_display_names
from eli5._feature_weights import get_top_features


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


@explain_prediction.register(XGBClassifier)
@singledispatch
def explain_prediction_xgboost(
        clf, doc,
        vec=None,
        top=None,
        target_names=None,
        targets=None,
        feature_names=None,
        vectorized=False):
    """ Return an explanation of XGBoost prediction (via scikit-learn wrapper).
    """
    vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    if feature_names.bias_name is None:
        # XGBoost estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'

    # TODO: do it properly (not sure how yet, also include bias handling?)
    fnames = feature_names.feature_names
    missing = FormattedFeatureName('Missing features')
    missing_idx = feature_names.n_features
    if isinstance(fnames, (list, np.ndarray)):
        fnames = list(fnames)
        fnames.append(missing)
    elif isinstance(fnames, dict):
        fnames = dict(fnames)
        fnames[missing_idx] = missing
    elif fnames is None:
        fnames = {missing_idx: missing}
    feature_names.feature_names = fnames
    feature_names.n_features += 1

    X = get_X(doc, vec, vectorized=vectorized)
    if sp.issparse(X):
        # Work around XGBoost issue:
        # https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        X = X.tocsc()

    proba = predict_proba(clf, X)

    display_names = get_target_display_names(
        clf.classes_, target_names, targets)

    scores_weights = prediction_feature_weights(clf, X, feature_names)

    res = Explanation(
        estimator=repr(clf),
        method='decision paths',
        targets=[],
    )
    if clf.n_classes_ > 2:
        for label_id, label in display_names:
            score, feature_weights = scores_weights[label_id]
            target_expl = TargetExplanation(
                target=label,
                feature_weights=get_top_features(
                    feature_names, feature_weights, top),
                score=score,
                proba=proba[label_id] if proba is not None else None,
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            res.targets.append(target_expl)
    else:
        (score, feature_weights), = scores_weights
        target_expl = TargetExplanation(
            target=display_names[1][1],
            feature_weights=get_top_features(
                feature_names, feature_weights, top),
            score=score,
            proba=proba[1] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        res.targets.append(target_expl)

    return res


def prediction_feature_weights(clf, X, feature_names):
    """ For each target, return score and numpy array with feature weights
    on this prediction, following an idea from
    http://blog.datadive.net/interpreting-random-forests/
    """
    # XGBClassifier does not have pred_leaf argument, so use booster
    booster = clf.booster()  # type: Booster
    leaf_ids, = booster.predict(DMatrix(X, missing=clf.missing), pred_leaf=True)
    tree_dumps = booster.get_dump(with_stats=True)
    assert len(tree_dumps) == len(leaf_ids)
    if clf.n_classes_ > 2:
        # For multiclass, XGBoost stores dumps and leaf_ids in a 1d array,
        # so we need to split them.
        scores_weights = [
            target_feature_weights(
                leaf_ids[class_idx::clf.n_classes_],
                tree_dumps[class_idx::clf.n_classes_],
                feature_names,
            ) for class_idx in range(clf.n_classes_)]
    else:
        scores_weights = [
            target_feature_weights(leaf_ids, tree_dumps, feature_names)]
    return scores_weights


def target_feature_weights(leaf_ids, tree_dumps, feature_names):
    feature_weights = np.zeros(len(feature_names))
    # All trees in XGBoost give equal contribution to the prediction:
    # it is equal to sum of "leaf" values in leafs
    # before applying loss-specific function
    # (e.g. logistic for "binary:logistic" loss).
    score = 0
    for text_dump, leaf_id in zip(tree_dumps, leaf_ids):
        leaf = indexed_leafs(parse_tree_dump(text_dump))[leaf_id]
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
            diff = child['leaf'] - node['leaf']
            res_map = {node[k]: k for k in ['yes', 'no', 'missing']}
            res = yn_res = res_map[child['nodeid']]
            if res == 'missing':
                if node['yes'] == node['missing']:
                    yn_res = 'no'
                elif node['no'] == node['missing']:
                    yn_res = 'yes'
            # Condition is "x < split_condition", so sign is inverted
            sign = {'yes': -1, 'no': 1}[yn_res]
            # Last feature is for all missing features
            idx = ((feature_names.n_features - 1) if res == 'missing'
                   else feature_idx)
            feature_weights[idx] += diff * sign
        # Root "leaf" value is interpreted as bias
        feature_weights[feature_names.bias_idx] += path[0]['leaf']
    return score, feature_weights


def indexed_leafs(parent):
    """ Return a leaf nodeid -> node dictionary with
    "parent" and "leaf" (average child "leaf" value) added to all nodes.
    """
    indexed = {}
    for child in parent['children']:
        child['parent'] = parent
        if 'leaf' in child:
            indexed[child['nodeid']] = child
        else:
            indexed.update(indexed_leafs(child))
    covers = np.array([child['cover'] for child in parent['children']])
    covers /= np.sum(covers)
    leafs = np.array([child['leaf'] for child in parent['children']])
    parent['leaf'] = np.mean(leafs * covers)
    return indexed


def parse_tree_dump(text_dump):
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
