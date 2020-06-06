# -*- coding: utf-8 -*-
"""
Credits
    * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
    * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
    * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
"""

from __future__ import absolute_import
from typing import Any, Optional, Tuple, List

import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer

from eli5.nn.gradcam import (
    _validate_targets,
    _validate_classification_target,
)


def gradcam_backend_keras(model, # type: Model
                          doc, # type: np.ndarray
                          targets, # type: Optional[List[int]]
                          activation_layer, # type: Layer
                          ):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    Compute the terms and by-products required by the Grad-CAM formula.

    Parameters
    ----------
    model : keras.models.Model
        Differentiable network.

    doc : numpy.ndarray
        Input to the network.

    targets : list, optional
        Index into the network's output,
        indicating the output node that will be
        used as the "loss" or scalar target during differentiation.

    activation_layer : keras.layers.Layer
        Keras layer instance to differentiate with respect to.


    See :func:`eli5.keras.explain_prediction` for description of the
    ``model``, ``doc``, and ``targets`` parameters.

    Returns
    -------
    (activations, gradients, predicted_idx, predicted_val) :
        (numpy.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Values of variables.
    """
    # class score
    predicted_idx, predicted_val = _classification_target(model, targets)

    # output of target activation layer, i.e. activation maps of a conv layer
    activation_output = activation_layer.output

    # score w.r.p.t. activation layer
    grads = _calc_gradient(predicted_val, [activation_output])

    # TODO: gradcam on input layer
    evaluate = K.function([model.input],
                          [activation_output,
                           grads,
                           predicted_val,
                           predicted_idx]
                          )

    # evaluate the graph (do actual computations)
    activations, grads, predicted_val, predicted_idx = evaluate([doc])
    return activations, grads, predicted_idx, predicted_val


def _calc_gradient(ys, xs):
    # (Any, list) -> Any
    # TODO: In the future we can replace the annotation Any with a tensor type in Keras backend
    """
    Return the gradient of scalar ``ys`` with respect to each of list ``xs``,
    (must be singleton)
    and apply gradient normalization.
    """
    # differentiate ys (scalar) with respect to each variable in xs
    # K.gradients tends to produce bigger values than tf.gradients
    # Note that gradients also point to the direction of the greatest increase in ys
    # Meaning that the class score is maximized if we move in the direction of the gradient.
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
                         'Check that the model is differentiable and try again. '
                         'ys: {}. xs: {}. grads: {}'.format(ys, xs, grads))

    # this seems to make the heatmap less noisy
    grads = K.l2_normalize(grads)
    return grads


def _classification_target(model, targets):
    # type: (Model, Optional[List[int]]) -> Tuple[Any, Any]
    """Get a predicted index and its value from a classification based model."""
    # TODO: maybe pass the loss/score to the gradcam function.
    # This would be consistent with what is done in
    # https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # and https://github.com/ramprs/grad-cam/blob/master/classification.lua
    if targets is not None:
        _validate_targets(targets)
        target, = targets
        _validate_classification_target(target, model.output_shape)
        # make a dummy index
        predicted_idx = K.constant([target], dtype='int64')
    else:
        # take the index with the highest value
        # from the array of predictions
        predicted_idx = K.argmax(model.output, axis=-1)

    # access value by index
    predicted_val = K.gather(model.output[0, :], predicted_idx)
    return predicted_idx, predicted_val
