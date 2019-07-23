# -*- coding: utf-8 -*-
import numpy as np # type: ignore


def gradcam_text_spans(heatmap, tokens, doc, pad_x, padding_type):
    # we resize before cutting off padding?
    # FIXME: might want to do this when formatting the explanation?
    heatmap = resize_1d(heatmap, tokens)

    if pad_x is not None:
        # remove padding
        tokens, heatmap = _trim_padding(pad_x, padding_type, doc,
                                        tokens, heatmap)
    document = _construct_document(tokens)
    spans = _build_spans(tokens, heatmap, document)
    weighted_spans = WeightedSpans([
        DocWeightedSpans(document, spans=spans)
    ]) # why list? - for each vectorized - don't need multiple vectorizers?
       # multiple highlights? - could do positive and negative expl?
    return tokens, heatmap, weighted_spans


def gradcam_heatmap(activations, grads, relu, counterfactual):
    if counterfactual:
        # negate grads for a "counterfactual explanation"
        # can equivalently negate ys loss scalar in gradcam_backend
        grads = -grads

    weights = compute_weights(grads)
    heatmap = get_localization_map(weights, activations, relu=relu)
    heatmap, = heatmap # FIXME: hardcode batch=1 for now
    return heatmap


def get_localization_map(weights, activations, relu=True):
    # type: (np.ndarray, np.ndarray, bool) -> np.ndarray
    """
    Generate a localization map (heatmap) using Gradient-weighted Class Activation Mapping 
    (Grad-CAM) (https://arxiv.org/pdf/1610.02391.pdf).
    
    The values for the parameters can be obtained from
    :func:`eli5.keras.gradcam.gradcam_backend`.

    Parameters
    ----------
    weights : numpy.ndarray
        Activation weights, vector with one weight per map, 
        rank 2 (batch size included).

    activations : numpy.ndarray
        Forward activation map values, vector of tensors, 
        rank n (batch size included).

    relu: boolean
        Whether to apply ReLU to the final heatmap (remove negatives).
    
    Returns
    -------
    lmap : numpy.ndarray
        A Grad-CAM localization map,
        with shape like the dimension portion of ``activations``.

    Notes
    -----
    We currently make two assumptions in this implementation
        * We are dealing with images as our input to ``model``.
        * We are doing a classification. ``model``'s output is a class scores or probabilities vector.

    Credits
        * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
        * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
        * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
    """
    # For reusability, this function should only use numpy operations
    # No library specific operations
    
    # we need to multiply (dim1, ..., dimn, maps,) by (maps,) over the dimension axes
    # and add each result to (dim1, ..., dimn) results array
    # there does not seem to be an easy way to do this:
    # see: https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    
    # activations shapes (channels last):
    # conv: (batch, ..., channels)
    # recurrent: (batch, timesteps, units)
    if len(activations.shape) == 2:
        # vector (with batch)
        lmap_shape = activations.shape
    else:
        # higher rank
        # last dimension is assumed to be the channels (for convnets)
        lmap_shape = activations.shape[:-1] # -> (batch, dim1, dim2)
    lmap = np.zeros(lmap_shape, dtype=np.float64)

    # weighted linear combination
    for activation_map, weight in _generate_maps_weights(activations, weights):
        # add result to the entire localization map (NOT pixel by pixel)
        lmap += activation_map * weight

    if relu:
        # apply ReLU
        lmap = np.maximum(lmap, 0) 
    return lmap


def _generate_maps_weights(activations, weights):
    """Yield tuples of (activation_map, weight) 
    from ``activations`` and ``weights``."""
    assert activations.shape[-1] == weights.shape[-1]
    num_maps = weights.shape[-1]
    return ((activations[..., i], weights[..., i]) 
                for i in range(num_maps))


def compute_weights(grads): # made public for transparency
    """
    Calculate weights, pooling over ``grads``.
    """
    # Global Average Pooling of gradients to get the weights
    # note that axes are in range [-rank(x), rank(x)) (we start from 1, not 0)
    
    # TEXT FIXME: conv1d vs conv2d (diff axis counts)
    # rank 0 = batch
    # rank 1 = dim
    # rank 2 ...
    # rank n = mapno
    # use np just because Keras backend / tensorflow is harder
    # https://stackoverflow.com/questions/48082900/in-tensorflow-what-is-the-argument-axis-in-the-function-tf-one-hot
    # https://medium.com/@aerinykim/tensorflow-101-what-does-it-mean-to-reduce-axis-9f39e5c6dea2
    # https://www.tensorflow.org/guide/tensors
    # weights = K.mean(grads, axis=(1, 2)) # +1 axis num because we have batch still?
    shape = [(axis_no, dim) for (axis_no, dim) in enumerate(grads.shape)]
    # ignore batch
    # ignore last (number of maps / channels)
    # FIXME: hardcoded shape
    pooling_axes = shape[1:-1]
    if len(pooling_axes) == 0:
        weights = grads # no need to average
    else:
        axes = [axis_no for (axis_no, dim) in pooling_axes]
        weights = np.mean(grads, axis=tuple(axes))
    return weights