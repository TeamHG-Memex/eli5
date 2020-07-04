# -*- coding: utf-8 -*-
from typing import Tuple, Generator
import numpy as np # type: ignore


DESCRIPTION_GRADCAM = """
Grad-CAM visualization for classification tasks; 
output is explanation object that contains a heatmap.
"""


def gradcam_heatmap(activations, grads, relu=True, counterfactual=False):
    # type: (np.ndarray, np.ndarray, bool, bool) -> np.ndarray
    """
    Build a Gradient-weighted Class Activation Mapping (Grad-CAM)
    (https://arxiv.org/pdf/1610.02391.pdf) heatmap using the required terms.

    See Grad-CAM "backends" to obtain the required terms.

    Parameters
    ----------
    activations : numpy.ndarray
        Activations of a target layer.

    grads : numpy.ndarray
        Gradients of output with respect to activations.

    relu : bool, optional
        Whether to apply ReLU to the produced heatmap.

        Default is `True`.

    counterfactual : bool, optional
        Whether to negate gradients to produce a counterfactual explanation.

        Default is `False`.

    Returns
    -------
    heatmap : numpy.ndarray
        The resulting heatmap as an array, shape of the spatial dimensions of activations.
        May have batch size.
    """
    if counterfactual:
        # negate grads for a "counterfactual explanation"
        # can equivalently negate ys loss scalar in gradcam_backend
        grads = -grads
    weights = compute_weights(grads)
    heatmap = get_localization_map(weights, activations, relu=relu)
    return heatmap


def get_localization_map(weights, activations, relu=True):
    # type: (np.ndarray, np.ndarray, bool) -> np.ndarray
    """
    Generate a localization map (heatmap) using weights and activations.

    Parameters
    ----------
    weights : numpy.ndarray
        Activation weights, vector with one weight per map,
        (batch size included).

    activations : numpy.ndarray
        Forward activation map values, vector of tensors,
        (batch size included).

    relu: boolean
        Whether to apply ReLU to the final heatmap (remove negatives).

    Returns
    -------
    lmap : numpy.ndarray
        A Grad-CAM localization map,
        with shape like the spatial portion of ``activations``.
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
        # give user warning if result is all 0's - might be something wrong
    return lmap


def _generate_maps_weights(activations, weights):
    # type: (np.ndarray, np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]
    """
    Yield tuples of (activation_map, weight)
    from ``activations`` and ``weights``,
    both with shape (batch, dim...).
    """
    assert activations.shape[-1] == weights.shape[-1]
    dims = len(activations.shape)
    if dims < 3:
        # (batch, dim1)
        # take as is, there are no "maps"
        # else we are setting all of heatmap to the same value 
        # by adding length 1 results
        # generator with a single item
        yield (activations, weights,)
    else:
        # (batch, dim1, ..., dimn, channels)
        # where activations and weights may have differing number of dims
        num_maps = weights.shape[-1]
        for i in range(num_maps):
            yield (activations[..., i], weights[..., i],)


def compute_weights(grads): # made public for transparency
    # type: (np.ndarray) -> np.ndarray
    """
    Calculate weights, global average pooling (taking the mean) over
    spatial dimensions of ``grads`` (axes between first and last).

    Parameters
    ----------

    grads : numpy.ndarray
        Gradients to pool.

    Returns
    -------

    weights : numpy.ndarray
        Gradients pooled.
    """
    shape = [(axis_idx, dim) for (axis_idx, dim) in enumerate(grads.shape)]
    pooling_axes = shape[1:-1]  # ignore batch and channels/last axis
    if len(pooling_axes) == 0:
        weights = grads  # no need to average
    else:
        axes = [axis_idx for (axis_idx, dim) in pooling_axes]
        weights = np.mean(grads, axis=tuple(axes))
        # weights = K.mean(grads, axis=(1, 2))     # alternative keras.backend/TF implementation
        # note that axes are in range [-rank(x), rank(x)) (we start from 1, not 0)
        # +1 axis num because we have batch still?
        # https://stackoverflow.com/questions/48082900/in-tensorflow-what-is-the-argument-axis-in-the-function-tf-one-hot
        # https://medium.com/@aerinykim/tensorflow-101-what-does-it-mean-to-reduce-axis-9f39e5c6dea2
        # https://www.tensorflow.org/guide/tensors
    return weights


def _validate_targets(targets):
    # type: (list) -> None
    """Check whether ``targets``, the targetted classes for Grad-CAM, 
    has correct type and values."""
    if not isinstance(targets, list):
        raise TypeError('Invalid argument "targets" (must be a list): %s' % targets)
    else:
        if len(targets) != 1:
            raise ValueError('More than one prediction target '
                             'is currently not supported ' 
                             '(found a list that is not length 1): '
                             '{}'.format(targets))
        else:
            target = targets[0]
            if not isinstance(target, int):
                raise TypeError('Prediction target must be int. '
                                'Got: {}'.format(target))


def _validate_classification_target(target, output_shape):
    # type: (int, Tuple[int, ...]) -> None
    """Check that ``target`` is a correct classification target
    into ``output_shape``, a tuple representing dimensions
    of the final output layer (including batch dimension)."""
    output_nodes = output_shape[1:][0]
    if not (0 <= target < output_nodes):
        raise ValueError('Prediction target index is ' 
                         'outside the required range [0, {}). '
                         'Got {}'.format(output_nodes, target))


def _validate_heatmap(heatmap):
    # type: (np.ndarray) -> None
    """Utility function to check that the ``heatmap``
    argument has the right type."""
    if not isinstance(heatmap, np.ndarray):
        raise TypeError('heatmap must be a numpy.ndarray instance. '
                        'Got: "{}" (type "{}").'.format(heatmap, type(heatmap)))