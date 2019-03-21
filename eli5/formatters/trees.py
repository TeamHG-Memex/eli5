# -*- coding: utf-8 -*-
from typing import Any

from eli5.base import TreeInfo, NodeInfo


def tree2text(tree_obj, indent=4):
    # type: (TreeInfo, int) -> str
    """
    Return text representation of a decision tree.
    """
    parts = []

    def _format_node(node, depth=0):
        # type: (NodeInfo, int) -> None
        def p(*args):
            # type: (*str) -> None
            parts.append(" " * depth * indent)
            parts.extend(args)

        if node.is_leaf:
            value_repr = _format_leaf_value(tree_obj, node)
            parts.append("  ---> {}".format(value_repr))
        else:
            assert node.left is not None
            assert node.right is not None
            feat_name = node.feature_name

            if depth > 0:
                parts.append("\n")
            left_samples = node.left.sample_ratio
            p("{feat_name} <= {threshold:0.3f}  ({left_samples:0.1%})".format(
                left_samples=left_samples,
                feat_name=feat_name,
                threshold=node.threshold,
            ))
            _format_node(node.left, depth=depth + 1)

            parts.append("\n")
            right_samples = node.right.sample_ratio
            p("{feat_name} > {threshold:0.3f}  ({right_samples:0.1%})".format(
                right_samples=right_samples,
                feat_name=feat_name,
                threshold=node.threshold,
                ))
            _format_node(node.right, depth=depth + 1)

    _format_node(tree_obj.tree)
    return "".join(parts)


def _format_leaf_value(tree_obj, node):
    # type: (...) -> str
    if tree_obj.is_classification:
        if len(node.value_ratio) == 2:
            return "{:0.3f}".format(node.value_ratio[1])
        else:
            return _format_array(node.value_ratio, "{:0.3f}")
    else:
        value = node.value
        if len(value) == 1:
            return "{}".format(value[0])
        else:
            assert all(len(v) == 1 for v in value)
            return _format_array([v[0] for v in value], "{}")


def _format_array(x, fmt):
    # type: (Any, str) -> str
    """
    >>> _format_array([0, 1.0], "{:0.3f}")
    '[0.000, 1.000]'
    """
    value_repr = ", ".join(fmt.format(v) for v in x)
    return "[{}]".format(value_repr)
