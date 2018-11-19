# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
from collections import defaultdict
from typing import DefaultDict, Optional

import numpy as np  # type: ignore
import lightgbm  # type: ignore

from eli5.explain import explain_weights, explain_prediction
from eli5._feature_importances import get_feature_importance_explanation
from eli5.sklearn.utils import handle_vec, get_X, get_X0, add_intercept, predict_proba
from eli5._decision_path import get_decision_path_explanation


DESCRIPTION_LIGHTGBM = """
LightGBM feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""


@explain_weights.register(lightgbm.LGBMClassifier)
@explain_weights.register(lightgbm.LGBMRegressor)
def explain_weights_lightgbm(lgb,
                             vec=None,
                             top=20,
                             target_names=None,  # ignored
                             targets=None,  # ignored
                             feature_names=None,
                             feature_re=None,
                             feature_filter=None,
                             importance_type='gain',
                             ):
    """
    Return an explanation of an LightGBM estimator (via scikit-learn wrapper
    LGBMClassifier or LGBMRegressor) as feature importances.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``feature_names``,
    ``feature_re`` and ``feature_filter`` parameters.

    ``target_names`` and ``targets`` parameters are ignored.
    
    Parameters
    ----------
    importance_type : str, optional
        A way to get feature importance. Possible values are:

        - 'gain' - the average gain of the feature when it is used in trees
          (default)
        - 'split' - the number of times a feature is used to split the data
          across all trees
        - 'weight' - the same as 'split', for compatibility with xgboost
    """
    coef = _get_lgb_feature_importances(lgb, importance_type)
    lgb_feature_names = lgb.booster_.feature_name()
    return get_feature_importance_explanation(lgb, vec, coef,
        feature_names=feature_names,
        estimator_feature_names=lgb_feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        description=DESCRIPTION_LIGHTGBM,
        num_features=coef.shape[-1],
        is_regression=isinstance(lgb, lightgbm.LGBMRegressor),
    )


@explain_prediction.register(lightgbm.LGBMClassifier)
@explain_prediction.register(lightgbm.LGBMRegressor)
def explain_prediction_lightgbm(
        lgb, doc,
        vec=None,
        top=None,
        top_targets=None,
        target_names=None,
        targets=None,
        feature_names=None,
        feature_re=None,
        feature_filter=None,
        vectorized=False,
        ):
    """ Return an explanation of LightGBM prediction (via scikit-learn wrapper
    LGBMClassifier or LGBMRegressor) as feature weights.

    See :func:`eli5.explain_prediction` for description of
    ``top``, ``top_targets``, ``target_names``, ``targets``,
    ``feature_names``, ``feature_re`` and ``feature_filter`` parameters.

    ``vec`` is a vectorizer instance used to transform
    raw features to the input of the estimator ``xgb``
    (e.g. a fitted CountVectorizer instance); you can pass it
    instead of ``feature_names``.

    ``vectorized`` is a flag which tells eli5 if ``doc`` should be
    passed through ``vec`` or not. By default it is False, meaning that
    if ``vec`` is not None, ``vec.transform([doc])`` is passed to the
    estimator. Set it to True if you're passing ``vec``,
    but ``doc`` is already vectorized.

    Method for determining feature importances follows an idea from
    http://blog.datadive.net/interpreting-random-forests/.
    Feature weights are calculated by following decision paths in trees
    of an ensemble.
    Each leaf has an output score, and expected scores can also be assigned
    to parent nodes.
    Contribution of one feature on the decision path is how much expected score
    changes from parent to child.
    Weights of all features sum to the output score of the estimator.
    """

    vec, feature_names = handle_vec(lgb, doc, vec, vectorized, feature_names)
    if feature_names.bias_name is None:
        # LightGBM estimators do not have an intercept, but here we interpret
        # them as having an intercept
        feature_names.bias_name = '<BIAS>'
    X = get_X(doc, vec, vectorized=vectorized)

    proba = predict_proba(lgb, X)
    weight_dicts = _get_prediction_feature_weights(lgb, X, _lgb_n_targets(lgb))
    x = get_X0(add_intercept(X))

    is_regression = isinstance(lgb, lightgbm.LGBMRegressor)
    is_multiclass = _lgb_n_targets(lgb) > 2
    names = lgb.classes_ if not is_regression else ['y']

    def get_score_weights(_label_id):
        _weights = _target_feature_weights(
            weight_dicts[_label_id],
            num_features=len(feature_names),
            bias_idx=feature_names.bias_idx,
        )
        _score = _get_score(weight_dicts[_label_id])
        return _score, _weights

    return get_decision_path_explanation(
        lgb, doc, vec,
        x=x,
        feature_names=feature_names,
        feature_filter=feature_filter,
        feature_re=feature_re,
        top=top,
        vectorized=vectorized,
        original_display_names=names,
        target_names=target_names,
        targets=targets,
        top_targets=top_targets,
        is_regression=is_regression,
        is_multiclass=is_multiclass,
        proba=proba,
        get_score_weights=get_score_weights,
     )


def _lgb_n_targets(lgb):
    if isinstance(lgb, lightgbm.LGBMClassifier):
        return lgb.n_classes_
    else:
        return 1


def _get_lgb_feature_importances(lgb, importance_type):
    aliases = {'weight': 'split'}
    coef = lgb.booster_.feature_importance(
        importance_type=aliases.get(importance_type, importance_type)
    )
    norm = coef.sum()
    return coef / norm if norm else coef


def _compute_node_values(tree_info):
    """ Add node_value key with an expected value for non-leaf nodes """
    def walk(tree):
        if 'leaf_value' in tree:
            return tree['leaf_value'], tree.get('leaf_count', 0)
        left_value, left_count = walk(tree['left_child'])
        right_value, right_count = walk(tree['right_child'])
        count = left_count + right_count
        if tree['split_gain'] <= 0:
            assert left_value == right_value
            tree['_node_value'] = left_value
        else:
            tree['_node_value'] = (left_value * left_count +
                                  right_value * right_count) / count
        return tree['_node_value'], count

    for tree in tree_info:
        walk(tree['tree_structure'])


def _get_decision_path(leaf_index, split_index, leaf_id):
    path, split_features = [], []
    parent_id, leaf = leaf_index[leaf_id]
    path.append(leaf['leaf_value'])
    while True:
        if parent_id == -1:
            break
        parent_id, node = split_index[parent_id]
        path.append(node['_node_value'])
        split_features.append(node['split_feature'])

    path.reverse()
    changes = _changes(path)
    bias, path = changes[0], list(zip(reversed(split_features), changes[1:]))
    return bias, path


def _changes(path):
    """
    >>> _changes([2, 3, 0, 5])
    [2, 1, -3, 5]
    >>> _changes([2])
    [2]
    """
    res = [path[0]]
    res += [p - p_prev for p, p_prev in zip(path[1:], path)]
    return res


def _get_leaf_split_indices(tree_structure):
    leaf_index = {}   # leaf id => (parent_id, leaf)
    split_index = {}  # split id => (parent_id, subtree)

    def walk(tree, parent_id=-1):
        if 'leaf_index' in tree:
            # regular leaf
            leaf_index[tree['leaf_index']] = (parent_id, tree)
        elif 'split_index' not in tree:
            # one-leaf tree producing a constant without splits
            leaf_index[0] = (parent_id, tree)
        else:
            # split node
            split_index[tree['split_index']] = (parent_id, tree)
            walk(tree['left_child'], tree['split_index'])
            walk(tree['right_child'], tree['split_index'])

    walk(tree_structure)
    return leaf_index, split_index


def _get_prediction_feature_weights(lgb, X, n_targets):
    """ 
    Return a list of {feat_id: value} dicts with feature weights, 
    following ideas from  http://blog.datadive.net/interpreting-random-forests/  
    """
    if n_targets == 2:
        n_targets = 1
    dump = lgb.booster_.dump_model()
    tree_info = dump['tree_info']
    _compute_node_values(tree_info)
    pred_leafs = lgb.booster_.predict(X, pred_leaf=True).reshape(-1, n_targets)
    tree_info = np.array(tree_info).reshape(-1, n_targets)
    assert pred_leafs.shape == tree_info.shape

    res = []
    for target in range(n_targets):
        feature_weights = defaultdict(float)  # type: DefaultDict[Optional[str], float]
        for info, leaf_id in zip(tree_info[:, target], pred_leafs[:, target]):
            leaf_index, split_index = _get_leaf_split_indices(
                info['tree_structure']
            )
            bias, path = _get_decision_path(leaf_index, split_index, leaf_id)
            feature_weights[None] += bias
            for feat, value in path:
                feature_weights[feat] += value
        res.append(dict(feature_weights))
    return res


def _target_feature_weights(feature_weights_dict, num_features, bias_idx):
    feature_weights = np.zeros(num_features)
    for k, v in feature_weights_dict.items():
        if k is None:
            feature_weights[bias_idx] = v
        else:
            feature_weights[k] = v
    return feature_weights


def _get_score(feature_weights_dict):
    return sum(feature_weights_dict.values())
