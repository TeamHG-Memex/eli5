# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Tuple, List

import numpy as np # type: ignore
import keras # type: ignore
import keras.backend as K # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Layer # type: ignore


def gradcam(weights, activations):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Generate a localization map (heatmap) using Gradient-weighted Class Activation Mapping 
    (Grad-CAM) (https://arxiv.org/pdf/1610.02391.pdf).
    
    The values for the parameters can be obtained from
    :func:`eli5.keras.gradcam.gradcam_backend`.

    Parameters
    ----------
    weights : numpy.ndarray
        Activation weights, vector with one weight per map, 
        rank 1.

    activations : numpy.ndarray
        Forward activation map values, vector of matrices, 
        rank 3.
    
    Returns
    -------
    lmap : numpy.ndarray
        A Grad-CAM localization map,
        rank 2, with values normalized in the interval [0, 1].

    Notes
    -----
    We currently make two assumptions in this implementation
        * We are dealing with images as our input to ``estimator``.
        * We are doing a classification. ``estimator``'s output is a class scores or probabilities vector.

    Credits
        * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
        * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
        * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
    """
    # For reusability, this function should only use numpy operations
    # Instead of backend library operations
    
    # Perform a weighted linear combination
    # we need to multiply (dim1, dim2, maps,) by (maps,) over the first two axes
    # and add each result to (dim1, dim2,) results array
    # there does not seem to be an easy way to do this:
    # see: https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    # spatial_shape = activations.shape[:2] # -> (dim1, dim2)
    spatial_shape = activations.shape[1:-1]
    lmap = np.zeros(spatial_shape, dtype=np.float64)

    # lmap = lmap[0, ...]
    weights = weights[0, ...]
    activations = activations[0, ...]

    # iterate through each activation map
    for i, w in enumerate(weights): 
        # weight * spatial map
        # add result to the entire localization map (NOT pixel by pixel)
        lmap += w * activations[..., i]

    lmap = np.maximum(lmap, 0) # ReLU

    # normalize lmap to [0, 1] ndarray
    # add eps to avoid division by zero in case lmap is 0's
    # this also means that lmap max will be slightly less than the 'true' max
    lmap = lmap / (np.max(lmap)+K.epsilon())
    return lmap


def compute_weights(grads): # made public for transparency
    """
    Calculate weights, pooling over ``grads``.
    """
    # Global Average Pooling of gradients to get the weights
    # note that axes are in range [-rank(x), rank(x)) (we start from 1, not 0)
    # TODO: decide whether this should go in gradcam_backend() or gradcam()
    
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
    # ignore last (number of maps)
    # FIXME: hardcoded shape
    pooling_axes = shape[1:-1]
    axes = [axis_no for (axis_no, dim) in pooling_axes]
    weights = np.mean(grads, axis=tuple(axes))
    return weights


def gradcam_backend(estimator, # type: Model
    doc, # type: np.ndarray
    targets, # type: Optional[List[int]]
    activation_layer # type: Layer
    ):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]
    """
    Compute the terms and by-products required by the Grad-CAM formula.
    
    Parameters
    ----------
    estimator : keras.models.Model
        Differentiable network.

    doc : numpy.ndarray
        Input to the network.

    targets : list, optional
        Index into the network's output,
        indicating the output node that will be
        used as the "loss" during differentiation.

    activation_layer : keras.layers.Layer
        Keras layer instance to differentiate with respect to.
    

    See :func:`eli5.keras.explain_prediction` for description of the 
    ``estimator``, ``doc``, ``targets`` parameters.

    Returns
    -------
    (weights, activations, gradients, predicted_idx, predicted_val) : (numpy.ndarray, ..., int, float)
        Values of variables.
    """
    # score for class in targets
    predicted_idx = _get_target_prediction(targets, estimator)
    predicted_val = K.gather(estimator.output[0,:], predicted_idx) # access value by index

    # output of target activation layer, i.e. activation maps of a convolutional layer
    activation_output = activation_layer.output 

    # score for class w.r.p.t. activation layer
    grads = _calc_gradient(predicted_val, [activation_output])

    evaluate = K.function([estimator.input], 
        [activation_output, grads, predicted_val, predicted_idx]
    )
    # evaluate the graph / do actual computations
    activations, grads, predicted_val, predicted_idx = evaluate([doc])

    # put into suitable form
    # FIXME: batch assumptions should be removed
    predicted_val = predicted_val[0]
    predicted_idx = predicted_idx[0]
    return activations, grads, predicted_idx, predicted_val


def _calc_gradient(ys, xs):
    # (K.variable, list) -> K.variable
    """
    Return the gradient of scalar ``ys`` with respect to each of list ``xs``,
    (must be singleton)
    and apply grad normalization.
    """
    # differentiate ys (scalar) with respect to each variable in xs
    grads = K.gradients(ys, xs)

    # grads gives a python list with a tensor (containing the derivatives) for each xs
    # to use grads with other operations and with K.function
    # we need to work with the actual tensors and not the python list
    grads, = grads # grads should be a singleton list (because xs is a singleton)

    # validate that the gradients were calculated successfully (no None's)
    # https://github.com/jacobgil/keras-grad-cam/issues/17#issuecomment-423057265
    # https://github.com/tensorflow/tensorflow/issues/783#issuecomment-175824168
    if grads is None:
        raise ValueError('Gradient calculation resulted in None values. '
                         'Check that the model is differentiate and try again. '
                         'ys: {}. xs: {}. grads: {}'.format(
                            ys, xs, grads))

    # this seems to make the heatmap less noisy
    grads =  K.l2_normalize(grads) 
    return grads


def _get_target_prediction(targets, estimator):
    # type: (Union[None, list], Model) -> K.variable
    """
    Get a prediction ID based on ``targets``, 
    from the model ``estimator`` (with a rank 2 tensor for its final layer).
    Returns a rank 1 K.variable tensor.
    """
    if isinstance(targets, list):
        # take the first prediction from the list
        if len(targets) == 1:
            target = targets[0]
            _validate_target(target, estimator.output_shape)
            predicted_idx = K.constant([target], dtype='int64')
        else:
            raise ValueError('More than one prediction target '
                             'is currently not supported ' 
                             '(found a list that is not length 1): '
                             '{}'.format(targets))
    elif targets is None:
        predicted_idx = K.argmax(estimator.output, axis=-1)
    else:
        raise TypeError('Invalid argument "targets" (must be list or None): %s' % targets)
    return predicted_idx


def _validate_target(target, output_shape):
    # type: (int, tuple) -> None
    """
    Check whether ``target``, 
    an integer index into the model's output
    is valid for the given ``output_shape``.
    """
    if isinstance(target, int):
        output_nodes = output_shape[1:][0]
        if not (0 <= target < output_nodes):
            raise ValueError('Prediction target index is ' 
                             'outside the required range [0, {}). ',
                             'Got {}'.format(output_nodes, target))
    else:
        raise TypeError('Prediction target must be int. '
                        'Got: {}'.format(target))