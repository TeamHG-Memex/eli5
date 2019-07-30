# -*- coding: utf-8 -*-
import numpy as np # type: ignore


DESCRIPTION_GRADCAM = """
Grad-CAM visualization for classification tasks; 
output is explanation object that contains a heatmap.
"""


def gradcam_heatmap(activations, grads, relu=True, counterfactual=False):
    # type: (np.ndarray, np.ndarray, bool, bool) -> np.ndarray
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
    
    # we need to multiply (batch, dim1, ..., dimn, maps,) by (batch, maps,) over the dimension axes
    # and add each result to (batch, dim1, ..., dimn) results array
    # there does not seem to be an easy way to do this:
    # see: https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    
    # shapes:
    # spatial: (batch, dim1, dim2, channels)
    # temporal: (batch, timesteps, units)
    # no shape: (batch, units)
    if len(activations.shape) == 2:
        # vector (with batch)
        lmap_shape = activations.shape
    else:
        # higher rank
        # last dimension is assumed to be the channels (for convnets)
        lmap_shape = activations.shape[:-1] # -> (batch, dim1, dim2)

    # initialize localization map to all 0's
    lmap = np.zeros(lmap_shape, dtype=np.float64)

    # take weighted linear combinations
    for activation_map, weight in _generate_maps_weights(activations, weights):
        # add result to the entire localization map (NOT pixel by pixel)
        combination = activation_map * weight
        lmap += combination

    if relu:
        # apply ReLU
        lmap = np.maximum(lmap, 0) 
    return lmap


def _generate_maps_weights(activations, weights):
    """
    Yield tuples of (activation_map, weight) 
    from ``activations`` and ``weights``,
    both with shape (batch, dim...).
    """
    assert activations.shape[-1] == weights.shape[-1]
    num_maps = weights.shape[-1]
    dims = len(activations.shape)
    if dims < 3:
        # (batch, dim1)
        # take as is, there are no "maps"
        # else we are setting all of heatmap to the same value 
        # by adding length 1 results
        # generator with a single item
        g = ((activations, weights),)
    else:
        # (batch, dim1, ..., dimn, channels)
        g = ((activations[..., i], weights[..., i]) for i in range(num_maps))
    return g


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