# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer, Lambda
from keras.preprocessing.image import load_img, img_to_array, array_to_img

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
    layer: valid target activation layer in the model to Grad-CAM on,
        one of: a valid keras layer name (str) or index (int), or layer instance (keras.layers.Layer)
        if None, get automatically.
    Returns: explanation object with the results in attributes.
    """
    explanation = Explanation(
        estimator.name, # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='Vanilla Grad-CAM',
        is_regression=False, # classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
    )
    activation_layer = get_activation_layer(estimator, layer)
    predicted = get_target_prediction(estimator, doc, targets)
    
    heatmap = grad_cam(estimator, doc, predicted, activation_layer)
    # TODO: consider renaming 'heatmap' to 'visualization'/'activations' (the output is not yet a heat map)

    # need to insert a 'channel' axis for a rank 3 image
    heatmap = np.expand_dims(heatmap, axis=-1)
    explanation.heatmap = array_to_img(heatmap)
    
    # take the single image from the input 'batch'
    doc = doc[0]
    explanation.image = array_to_img(doc)

    # TODO: return arrays, not images (arrays are more general)
    # Consider returning the resized version of the heatmap, just a grayscale array
    return explanation


def get_activation_layer(estimator, layer):
    """
    Return instance of the desired layer in the model.
    estimator: model whose layer is to be gotten
    layer: one of: integer index, string name of the layer, 
    callable that returns True if a layer instance matches.
    """        
    if layer is None:
        # Automatically get the layer if not provided
        # this might not be a good idea from transparency / user point of view
        # layer = get_last_activation_maps
        activation_layer = get_last_activation_layer(estimator)
    elif isinstance(layer, Layer):
        activation_layer = layer
    elif isinstance(layer, int):
        # conv_output =  [l for l in model.layers if l.name is layer_name]
        # conv_output = conv_output[0]
        # bottom-up horizontal graph traversal
        activation_layer = estimator.get_layer(index=layer)
        # These can raise ValueError if the layer index / name specified is not found
    elif isinstance(layer, str):
        activation_layer = estimator.get_layer(name=layer)

    # TODO: check activation_layer dimensions (is it possible to perform Grad-CAM on it?)
    return activation_layer


def get_last_activation_layer(estimator):
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
    # TODO: take in a single target as well, not just a list
    # does it make sense to take a list of targets. You can only Grad-CAM a single target?
    # TODO: consider changing signature / types for explain_prediction generic function
    # TODO: need to find a way to show the label for the passed prediction as well as its probability
    # TODO: multiple predictions list (keras-vis)
    if isinstance(targets, list):
        # take the first prediction from the list
        # TODO: validate list contents
        predicted_idx = targets[0]
    elif targets is None:
        predictions = model.predict(x)
        # FIXME: non-classification tasks
        predicted_idx = np.argmax(predictions)
    else:
        raise ValueError('Invalid argument "targets" (must be list or None): %s' % targets)
    return predicted_idx


def grad_cam(estimator, image, prediction_index, activation_layer):
    #
    # Credits:
    # Jacob Gildenblat for "https://github.com/jacobgil/keras-grad-cam",
    # Author of "https://github.com/PowerOfCreation/keras-grad-cam" for fixes to above.
    # Kotikalapudi, Raghavendra and contributors for "https://github.com/raghakot/keras-vis"
    # 

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
    """Calculate all the terms required by Grad-CAM 
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


def array_from_path(img_path, image_shape=None):
    """
    Utility method for loading input images from disk / path.
    Returns a single image as an array for an estimator's input
    img: one of: path to a single image file, PIL Image object, numpy array
    estimator: model instance, for resizing the image to the required input dimensions
    """
    # TODO: Take in PIL image object, or an array can also be multiple images.
    # "pipeline": path str -> PIL image -> numpy array
    im = load_img(img_path, target_size=image_shape)
    x = img_to_array(im)

    # we need to insert an axis at the 0th position to indicate the batch size
    # this is required by the keras predict() function
    x = np.expand_dims(x, axis=0)
    return x