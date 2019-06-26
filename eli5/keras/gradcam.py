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
    Generate a heatmap using Gradient-weighted Class Activation Mapping 
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
    heatmap : numpy.ndarray
        A Grad-CAM localization map,
        rank 2, with values normalized in the interval [0, 1].

    Notes
    -----
    We currently make two assumptions in this implementation
        * We are dealing with images as our input to ``estimator``.
        * We are doing a classification. Our ``estimator``'s output is a class scores vector.

    Credits
        * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
        * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
        * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
    """
    # Get required terms

    # For reusability, this function should only use numpy operations
    # Instead of backend library operations

    # Perform a weighted linear combination
    spatial_shape = activations.shape[:2]
    lmap = np.zeros(spatial_shape, dtype=np.float64)
    for i, w in enumerate(weights):
        # weight (for one activation map) * single activation map
        # add to the entire map (linear combination), NOT pixel by pixel
        lmap += w * activations[..., i]
        # TODO: can this be expressed in terms of numpy operations?

    lmap = np.maximum(lmap, 0) # ReLU

    # normalize lmap to [0, 1] ndarray
    # add eps to avoid division by zero in case lmap is 0's
    # this also means that lmap max will be slightly less than the 'true' max
    lmap = lmap / (np.max(lmap)+K.epsilon())
    return lmap


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
    (weights, activations, gradients, predicted_idx, score) : (numpy.ndarray, ..., int, float)
        Values of variables.
    """
    output = estimator.output
    predicted_idx = _get_target_prediction(targets, output)
    score = K.gather(output[0,:], predicted_idx) # access value by index

    # output of target layer, i.e. activation maps of a convolutional layer
    activation_output = activation_layer.output 

    # differentiate ys (scalar) with respect to each of xs (python list of variables)
    grads = K.gradients(score, [activation_output])
    # FIXME: this might have issues
    # See https://github.com/jacobgil/keras-grad-cam/issues/17
    # a fix is the following piece of code:
    # grads = [grad if grad is not None else K.zeros_like(var) 
    #         for (var, grad) in zip(xs, grads)]

    # grads gives a python list with a tensor (containing the derivatives) for each xs
    # to use grads with other operations and with K.function
    # we need to work with the actual tensors and not the python list
    grads, = grads # grads should be a singleton list (because xs is a singleton)
    grads =  K.l2_normalize(grads) # this seems to make the heatmap less noisy

    # Global Average Pooling of gradients to get the weights
    # note that axes are in range [-rank(x), rank(x)) (we start from 1, not 0)
    # TODO: decide whether this should go in gradcam_backend() or gradcam()
    weights = K.mean(grads, axis=(1, 2))

    evaluate = K.function([estimator.input], [weights, activation_output, grads, output, score, predicted_idx])
    # evaluate the graph / do actual computations
    weights, activations, grads, output, score, predicted_idx = evaluate([doc])
    
    # put into suitable form
    weights = weights[0]
    score = score[0]
    predicted_idx = predicted_idx[0]
    activations = activations[0, ...]
    grads = grads[0, ...]
    return weights, activations, grads, predicted_idx, score


def _get_target_prediction(targets, output):
    # type: (Union[None, list], K.variable) -> K.variable
    """
    Get a prediction ID, an index into the final layer 
    of the model ``output`` (rank 2 tensor), using ``targets``.
    Returns a rank 1 K.variable tensor.
    """
    # TODO: take in a single target as well, not just a list, 
    # consider changing signature / types for explain_prediction generic function
    # TODO: need to find a way to show the label for the passed prediction 
    # as well as its probability

    # FIXME: this is hard to test, as output must be evaluated first

    # TODO: maybe do the sum / loss in this function instead of gradcam. Return a tensor.
    # This would be consistent with what is done in https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # https://github.com/ramprs/grad-cam/blob/master/classification.lua
    # https://github.com/torch/nn/blob/master/doc/module.md
    if isinstance(targets, list):
        # take the first prediction from the list
        if len(targets) == 1:
            predicted_idx = targets[0]
            # TODO: validate list contents
            predicted_idx = K.constant([predicted_idx], dtype='int64')
        else:
            raise ValueError('More than one prediction target'
                             'is currently not supported' 
                             '(found a list that is not length 1):'
                             '{}'.format(targets))
            # TODO: use all predictions in the list
    elif targets is None:
        predicted_idx = K.argmax(output, axis=-1)
    else:
        raise TypeError('Invalid argument "targets" (must be list or None): %s' % targets)
        # TODO: in the future, accept different ways to specify target
        # label (str), float (in regression tasks), int (not a list) etc.
    return predicted_idx