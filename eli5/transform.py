"""Handling transformation pipelines in explanations"""

from singledispatch import singledispatch


@singledispatch
def transform_feature_names(transformer, in_names=None):
    if hasattr(transformer, 'get_feature_names'):
        return transformer.get_feature_names()
    raise NotImplementedError('transform_feature_names not available for '
                              '{}'.format(transformer))
