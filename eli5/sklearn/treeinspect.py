# -*- coding: utf-8 -*-
"""
Inspect scikit-learn decision trees.

This is an alternative to sklearn.tree.export which doesn't require graphviz
and provides a way to output result in text-based format.
"""
from __future__ import absolute_import, division
import os
import shutil
from tempfile import mkdtemp

from sklearn.tree import _tree, export_graphviz

from eli5.base import TreeInfo, NodeInfo


def get_tree_info(decision_tree, feature_names=None, **export_graphviz_kwargs):
    """
    Convert DecisionTreeClassifier or DecisionTreeRegressor
    to an inspectable object.
    """
    return TreeInfo(
        criterion=decision_tree.criterion,
        tree=_get_root_node_info(decision_tree, feature_names),
        graphviz=tree2dot(decision_tree,
                          feature_names=feature_names,
                          **export_graphviz_kwargs),
    )


def tree2dot(decision_tree, **export_graphviz_kwargs):
    # use temp files to support scikit-learn < 0.18
    path = mkdtemp()
    try:
        tmpfile = os.path.join(path, 'tree.dot')
        export_graphviz(decision_tree, tmpfile, **export_graphviz_kwargs)
        with open(tmpfile, 'rb') as f:
            return f.read().decode('utf8')
    finally:
        shutil.rmtree(path)


def _get_root_node_info(decision_tree, feature_names=None):
    res = _get_node_info(decision_tree.tree_, 0)
    _add_feature_names(res, feature_names)
    return res


def _add_feature_names(root, feature_names=None):
    for node in _treeiter(root):
        if not node.is_leaf:
            feat_id = node.feature_id
            if feature_names is None:
                node.feature_name = "x%s" % feat_id
            else:
                node.feature_name = feature_names[feat_id]


def _get_node_info(tree, node_id):
    is_leaf = tree.children_left[node_id] == _tree.TREE_LEAF
    value = _node_value(tree, node_id)
    node = NodeInfo(
        id=node_id,
        is_leaf=is_leaf,
        value=list(value),
        value_ratio=list(value / value.sum()),
        impurity=tree.impurity[node_id],
        samples=tree.n_node_samples[node_id],
        sample_ratio=tree.n_node_samples[node_id] / tree.n_node_samples[0],
    )
    if not is_leaf:
        node.feature_id = tree.feature[node_id]
        node.threshold = tree.threshold[node_id]
        node.left = _get_node_info(tree, tree.children_left[node_id])
        node.right = _get_node_info(tree, tree.children_right[node_id])
    return node


def _node_value(tree, node_id):
    if tree.n_outputs == 1:
        return tree.value[node_id][0, :]
    else:
        return tree.value[node_id]


def _treeiter(node):
    yield node
    if not node.is_leaf:
        for n in _treeiter(node.left):
            yield n
        for n in _treeiter(node.right):
            yield n
