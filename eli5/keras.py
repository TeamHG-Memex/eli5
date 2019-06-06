# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

import numpy as np
import keras
import keras.backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Layer, Lambda

from eli5.base import Explanation
from eli5.explain import explain_prediction


# note that Sequential subclasses Model, so we can just register the Model type
# Model subclasses Network, but is using Network with this function valid?
@explain_prediction.register(Model)
def explain_prediction_keras(estimator, doc, # model, image
                             target_names=None, # rename / provide prediction labels
                             targets=None, # prediction(s) to focus on, if None take top prediction
                             # new parameters:
                             layer=None, # which layer to focus on, 
                            ):
    """Explain prediction of a Keras model
    doc : image, 
        must be an input acceptable by the estimator,
        (see other functions for loading/preprocessing).
    targets: predictions
        a list of predictions
        integer for ImageNet classification
    layer: valid target layer in the model to Grad-CAM on,
        one of: a valid keras layer name (str) or index (int), or layer instance (keras.layers.Layer)
    Returns: explanation object with the results in attributes.
    """
    explanation = Explanation(
        estimator.name, # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='gradcam',
        is_regression=False, # classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
    )
    # TODO: grad-cam on multiple layers by passing a list of layers
    if layer is None:
        # Automatically get the layer if not provided
        # this might not be a good idea from transparency / user point of view
        # layer = get_last_activation_maps
        layer = -4
    target_layer = get_target_layer(estimator, layer)

    # get prediction to focus on
    if targets is None:
        predicted = get_target_prediction(estimator, doc)
    else:
        predicted = targets[0]
        # TODO: take in a single target as well, not just a list
        # does it make sense to take a list of targets. You can only Grad-CAM a single target?
        # TODO: consider changing signature / types for explain_prediction generic function
        # TODO: need to find a way to show the label for the passed prediction as well as its probability
        # TODO: multiple predictions list (keras-vis)

    heatmap = grad_cam(estimator, doc, predicted, target_layer)
    # TODO: consider renaming 'heatmap' to 'visualization'/'activations' (the output is not yet a heat map)

    # need to insert a 'channel' axis for a rank 3 image
    heatmap = np.expand_dims(heatmap, axis=-1)
    explanation.heatmap = image.array_to_img(heatmap)
    
    # take the single image from the input 'batch'
    doc = doc[0]
    explanation.image = image.array_to_img(doc)

    # TODO: return arrays, not images (arrays are more general)
    # Consider returning the resized version of the heatmap, just a grayscale array
    return explanation


def get_target_prediction(model, x, decoder=None):
    predictions = model.predict(x)
    # FIXME: non-classification tasks
    maximum_predicted_idx = np.argmax(predictions)
    return maximum_predicted_idx


def get_target_layer(estimator, desired_layer):
    """
    Return instance of the desired layer in the model.
    estimator: model whose layer is to be gotten
    desired_layer: one of: integer index, string name of the layer, 
    callable that returns True if a layer instance matches.
    """
    # TODO: Consider moving this to Keras utils, 
    # only passing an instance of Keras Layer to the `layer` argument.
    if isinstance(desired_layer, Layer):
        target_layer = desired_layer
    elif isinstance(desired_layer, int):
        # conv_output =  [l for l in model.layers if l.name is layer_name]
        # conv_output = conv_output[0]
        # bottom-up horizontal graph traversal
        target_layer = estimator.get_layer(index=desired_layer)
        # These can raise ValueError if the layer index / name specified is not found
    elif isinstance(desired_layer, str):
        target_layer = estimator.get_layer(name=desired_layer)

    # TODO: check target_layer dimensions (is it possible to perform Grad-CAM on it?)
    return target_layer


def get_last_activation_maps(estimator):
    # TODO: automatically get last Conv layer if layer_name and layer_index are None
    # Some ideas:
    # 1. look at layer name, exclude things like "softmax", "global average pooling", 
    # etc, include things like "conv" (but watch for false positives)
    # 2. look at layer input/output dimensions, to ensure they match
    return True


def grad_cam(estimator, image, prediction_index, target_layer):
    # FIXME: this assumes that we are doing classification
    # also we make the explicit assumption that we are dealing with images
    nb_classes = estimator.output_shape[1] # TODO: test this

    model = estimator
    output = estimator.output
    loss = K.sum(output[:, prediction_index])

    # we need to access the output attribute, else we get a TypeError: Failed to convert object to tensor
    target_output = target_layer.output # output of target layer, i.e. activation maps of a convolutional layer
    
    grads = K.gradients(loss, [target_output])
    # grads = [grad if grad is not None else K.zeros_like(var) 
    #         for (var, grad) in zip(xs, grads)]
    # https://github.com/jacobgil/keras-grad-cam/issues/17
    grads = grads[0]
    grads =  K.l2_normalize(grads) # this seems to produce less noise

    evaluate = K.function([model.input], [target_output, grads])

    target_output_val, grads_val = evaluate([image]) # do work
    target_output_val = target_output_val[0, ...]
    grads_val = grads_val[0, ...]

    weights = np.mean(grads_val, axis=(0, 1)) # Global Average Pooling
    # TODO: replace numpy operations with keras backend operations, i.e. K.mean

    # weighted linear combination
    spatial_shape = target_output_val.shape[:2]
    lmap = np.zeros(spatial_shape, dtype=np.float32)
    for i, w in enumerate(weights):
        # weight * single activation map
        # add to the entire map (linear combination), NOT pixel by pixel
        lmap += w * target_output_val[..., i]

    lmap = np.maximum(lmap, 0) # ReLU

    lmap = lmap / np.max(lmap) # -> [0, 1] ndarray
    return lmap


#
# Credits:
# Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam",
# Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to above.
# Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis"
# 