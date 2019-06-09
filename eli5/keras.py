# -*- coding: utf-8 -*-
"""Keras neural network explanations"""
from __future__ import absolute_import

import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer, Lambda
from keras.preprocessing.image import load_img, img_to_array, array_to_img

from eli5.base import Explanation
from eli5.explain import explain_prediction


DESCRIPTION_KERAS = """Grad-CAM visualization for image input; output is images"""

# note that Sequential subclasses Model, so we can just register the Model type
@explain_prediction.register(Model)
def explain_prediction_keras(estimator, doc, # model, image
                             target_names=None, # TODO: rename / provide prediction labels
                             targets=None, # prediction(s) to focus on, if None take top prediction
                             # new parameters:
                             layer=None, # which layer to focus on, 
                            ):
    """ Return an explanation of a Keras model for image input.

    Parameters
    ----------
    doc : tensor, required
        an input image acceptable by the estimator.
    targets: list or None, optional
        a list of prediction id's to focus on,
        (currently only the first prediction from the list is explained),
        If None, the model is fed the input and the top prediction is taken automatically.
    layer: int, str, or keras.layers.Layer instance, optional
        an activation layer in the model to perform Grad-CAM on, either a valid keras layer name (str) 
        a layer index (int), or an instance of keras.layers.Layer.
        If None, a suitable layer is attempted to be retrieved.
    """
    activation_layer = get_activation_layer(estimator, layer)
    predicted = get_target_prediction(estimator, doc, targets)
    
    heatmap = grad_cam(estimator, doc, predicted, activation_layer)
    # TODO: consider renaming 'heatmap' to 'visualization'/'activations' (the output is not yet a heat map)

    # need to insert a 'channel' axis for a rank 3 image
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = array_to_img(heatmap)
    
    # take the single image from the input 'batch'
    doc = doc[0]
    image = array_to_img(doc)

    # TODO: return arrays, not images (arrays are more general)
    # Consider returning the resized version of the heatmap, just a grayscale array
    return Explanation(
        estimator.name, # might want to replace this with something else, eg: estimator.summary()
        description=DESCRIPTION_KERAS,
        error='',
        method='Vanilla Grad-CAM',
        is_regression=False, # classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
        image=image,
        heatmap=heatmap,
    )


def get_activation_layer(estimator, layer):
    """
    Return instance of the desired layer in the model.
    See documentation of explain_prediction_keras for description of layer.
    Raises ValueError if layer is of wrong type.
    """        
    if layer is None:
        # Automatically get the layer if not provided
        # this might not be a good idea from transparency / user point of view
        # layer = get_last_activation_maps
        activation_layer = get_last_activation_layer(estimator)
    elif isinstance(layer, Layer):
        activation_layer = layer
    elif isinstance(layer, int):
        # bottom-up horizontal graph traversal
        activation_layer = estimator.get_layer(index=layer)
        # These can raise ValueError if the layer index / name specified is not found
    elif isinstance(layer, str):
        activation_layer = estimator.get_layer(name=layer)
    else:
        raise ValueError('Invalid layer (must be str, int, keras.layers.Layer, or None): %s' % layer)

    # TODO: check activation_layer dimensions (is it possible to perform Grad-CAM on it?)
    return activation_layer


def get_last_activation_layer(estimator):
    """Return a matching layer instance, searching backwards from model output.
    Raises ValueError if matching layer can not be found."""
    # we assume this is a simple feedforward network
    # linear search in reverse
    i = len(estimator.layers)-1
    while -1 < i and not is_suitable_activation_layer(estimator, i):
        i -= 1
    if -1 < i:
        # found a suitable layer
        return estimator.get_layer(index=i)
    else:
        raise ValueError('Could not find a suitable target layer automatically.')


def is_suitable_activation_layer(estimator, i):
    """Return True if layer at index i matches what is required by estimator for an activation layer."""
    # TODO: experiment with this, using many models and images, to find what works best
    # Some ideas: 
    # check layer type, i.e.: isinstance(l, keras.layers.Conv2D)
    # check layer name
    l = estimator.get_layer(index=i)
    # a check that asks 'can we resize this activation layer over the image?'
    rank = len(l.output_shape)
    required_rank = len(estimator.input_shape)
    return rank == required_rank


def get_target_prediction(model, x, targets):
    """Return a prediction ID. See documentation of explain_prediction_keras for explanation of targets"""
    # TODO: take in a single target as well, not just a list, consider changing signature / types for explain_prediction generic function
    # TODO: need to find a way to show the label for the passed prediction as well as its probability
    # TODO: multiple predictions list (keras-vis)

    # TODO: maybe do the sum / loss in this function instead of grad_cam. Return a tensor.
    # This would be consistent with what is done in https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # https://github.com/ramprs/grad-cam/blob/master/classification.lua
    # https://github.com/torch/nn/blob/master/doc/module.md
    if isinstance(targets, list):
        # take the first prediction from the list
        # TODO: validate list contents
        predicted_idx = targets[0]
    elif targets is None:
        predictions = model.predict(x)
        predicted_idx = np.argmax(predictions)
    else:
        raise ValueError('Invalid argument "targets" (must be list or None): %s' % targets)
    return predicted_idx


def grad_cam(estimator, image, prediction_index, activation_layer):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM),
    https://arxiv.org/pdf/1610.02391.pdf.

    Credits:
    * Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam".
    * Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to Jacob's implementation.
    * Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis".
    """
    # FIXME: this assumes that we are doing classification
    # also we make the explicit assumption that we are dealing with images

    weights, activations, grads_val = grad_cam_backend(estimator, image, prediction_index, activation_layer)

    # weighted linear combination
    spatial_shape = activations.shape[:2]
    lmap = np.zeros(spatial_shape, dtype=np.float32)
    for i, w in enumerate(weights):
        # weight * single activation map
        # add to the entire map (linear combination), NOT pixel by pixel
        lmap += w * activations[..., i]

    lmap = np.maximum(lmap, 0) # ReLU

    lmap = lmap / np.max(lmap) # -> [0, 1] ndarray
    return lmap


def grad_cam_backend(estimator, image, prediction_index, activation_layer):
    """Calculate terms required by Grad-CAM formula
    - weights, activation layer outputs, gradients"""
    output = estimator.output
    score = output[:, prediction_index]
    activation_output = activation_layer.output # output of target layer, i.e. activation maps of a convolutional layer
    
    grads = K.gradients(score, [activation_output])
    # grads = [grad if grad is not None else K.zeros_like(var) 
    #         for (var, grad) in zip(xs, grads)]
    # https://github.com/jacobgil/keras-grad-cam/issues/17
    grads = grads[0]
    grads =  K.l2_normalize(grads) # this seems to make the heatmap less noisy
    evaluate = K.function([estimator.input], [activation_output, grads])

    activations, grads_val = evaluate([image]) # do work
    activations = activations[0, ...]
    grads_val = grads_val[0, ...]
    weights = np.mean(grads_val, axis=(0, 1)) # Global Average Pooling
    # TODO: replace numpy operations with keras backend operations, i.e. K.mean
    return weights, activations, grads_val


def image_from_path(img_path, image_shape=None):
    """
    Utility method for loading a single images from disk / path.
    Returns a tensor suitable for an estimator's input.

    Parameters
    ----------
    img_path: str, required: 
        path to a single image file.
    image_shape: tuple, optional:
        A tuple of shape (height, width) that the image is to be resized to.
        For example, the required spatial input for the estimator.
        If None, no resizing is done.
    estimator: model instance, for resizing the image to the required input dimensions
    """
    # TODO: Take in PIL image object, or an array; "pipeline": path str -> PIL image -> numpy array
    # TODO: multiple images.
    im = load_img(img_path, target_size=image_shape)
    x = img_to_array(im)

    # we need to insert an axis at the 0th position to indicate the batch size (required by the model's input)
    x = np.expand_dims(x, axis=0)
    return x