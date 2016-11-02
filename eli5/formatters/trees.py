# -*- coding: utf-8 -*-


def tree2text(treedict, indent=4):
    """
    Return text representation of a decision tree.
    """
    parts = []

    def _format_node(node, depth=0):
        def p(*args):
            parts.append(" " * depth * indent)
            parts.extend(args)

        if node.is_leaf:
            value = node.value_ratio
            value_repr = ", ".join("{:0.3f}".format(v) for v in value)
            parts.append("  ---> [{value}]".format(value=value_repr))
        else:
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

    _format_node(treedict.tree)
    return "".join(parts)
