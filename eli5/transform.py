"""Handling transformation pipelines in explanations"""

from eli5.base_utils import singledispatch


@singledispatch
def transform_feature_names(transformer, in_names=None):
    """Get feature names for transformer output as a function of input names.

    Used by :func:`explain_weights` when applied to a scikit-learn Pipeline,
    this ``singledispatch`` should be registered with custom name
    transformations for each class of transformer.
    
    If there is no ``singledispatch`` handler registered for a transformer 
    class, ``transformer.get_feature_names()`` method is called; if there is
    no such method then feature names are not supported and 
    this function raises an exception.

    Parameters
    ----------
    transformer : scikit-learn-compatible transformer
    in_names : list of str, optional
        Names for features input to transformer.transform().
        If not provided, the implementation may generate default feature names
        if the number of input features is known.

    Returns
    -------
    feature_names : list of str
    """
    if hasattr(transformer, 'get_feature_names'):
        return transformer.get_feature_names()
    raise NotImplementedError('transform_feature_names not available for '
                              '{}'.format(transformer))
