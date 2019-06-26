# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, Tuple, List

import numpy as np # type: ignore
import keras # type: ignore
import keras.backend as K # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Layer, Lambda # type: ignore

from eli5.base import Explanation
from eli5.explain import explain_prediction


DESCRIPTION_KERAS = """Grad-CAM visualization for image classification; output is explanation
object that contains input image and heatmap image."""

# note that keras.models.Sequential subclasses keras.models.Model, so we can just register Model
@explain_prediction.register(Model)
def explain_prediction_keras(estimator, # type: Model
                             doc, # type: np.ndarray
                             target_names=None,
                             targets=None, # type: Optional[list]
                             layer=None, # type: Optional[Union[int, str, Layer]]
                            ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras image classifier.

    See :func:`eli5.explain_prediction` for more information about the ``estimator``,
    ``doc``, ``target_names``, and ``targets`` parameters.

    Parameters
    ----------
    estimator : keras.models.Model
        Instance of a Keras neural network model.

    doc : numpy.ndarray
        An input image as a tensor to ``estimator``, for example a ``numpy.ndarray``.

        The tensor must be of suitable shape for the ``estimator``. 

        For example, some models require input images to be 
        rank 4 in format `(batch_size, dims, ..., channels)` (channels last)
        or `(batch_size, channels, dims, ...)` (channels first), 
        where `dims` is usually in order `height, width`
        and `batch_size` is 1 for a single image.

        Check ``estimator.input_shape`` to confirm the required dimensions of the input tensor.

    target_names : list, optional
        *Not Implemented*. 

        Names for classes in the final output layer.

    targets : list[int], optional
        Prediction ID's to focus on.

        *Currently only the first prediction from the list is explained*. 
        The list must be length one.

        If None, the model is fed the input image and its top prediction 
        is taken as the target automatically.

    layer : int or str or keras.layers.Layer, optional
        The activation layer in the model to perform Grad-CAM on,
        a valid keras layer name, layer index, or an instance of a Keras layer.
        
        If None, a suitable layer is attempted to be retrieved. 
        See :func:`eli5.keras.search_layer_backwards` for details.

    Returns
    -------
    expl : Explanation
        A :class:`eli5.base.Explanation` object 
        with the ``image`` and ``heatmap`` attributes set.
    """
    # TODO: implement target_names
    # FIXME: Could doc be a Tensorflow object, not just a numpy array?
    validate_doc(estimator, doc)
    activation_layer = get_activation_layer(estimator, layer)
    
    heatmap, predicted_idx, score = grad_cam(estimator, doc, targets, activation_layer)
    # TODO: consider renaming 'heatmap' to 'visualization'/'activations'
    # (the output is not yet a heat map)

    print('Predicted class: %d' % predicted_idx)
    print('With probability: %f' % score)

    # TODO: consider passing multiple images in doc to perform grad-cam on multiple images
    doc = doc[0] # rank 4 batch -> rank 3 single image
    image = keras.preprocessing.image.array_to_img(doc) # -> PIL image

    return Explanation(
        estimator.name, # might want to replace this with something else, eg: estimator.summary()
        description=DESCRIPTION_KERAS,
        error='',
        method='Grad-CAM',
        is_regression=False, # TODO: classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
        image=image, # PIL image
        heatmap=heatmap, # 2D [0, 1] numpy array
    )


def validate_doc(estimator, doc):
    # type: (Model, np.ndarray) -> None
    """
    Check that input ``doc`` is suitable for ``estimator``.

    We assume that ``doc`` is an image.

    Parameters
    ----------
    estimator : keras.models.Model
        The model that will be fed the input.
        See :func:`explain_prediction_keras`.

    doc : numpy.ndarray
        The input to be checked.


    :raises ValueError: if ``doc`` shape does not match.
    """
    input_sh = estimator.input_shape
    doc_sh = doc.shape
    if len(input_sh) == 4:
        # rank 4 with (batch, ...) shape
        # check that we have only one image (batch size 1)
        single_batch = (1,) + input_sh[1:]
        if doc_sh != single_batch:
            raise ValueError('Batch size does not match (must be 1). ' 
                             'doc must be of shape: {}, '
                             'got: {}'.format(single_batch, doc_sh))
    else:
        # other shapes
        if doc_sh != input_sh:
            raise ValueError('Input and doc shapes do not match.'
                             'input: {}, doc: {}'.format(input_sh, doc_sh))
    # TODO: might want to just show a warning and attempt execution anyways?
    # TODO: check for TypeError as well


def get_activation_layer(estimator, layer):
    # type: (Model, Union[None, int, str, Layer]) -> Layer
    """
    Get an instance of the desired activation layer.
    
    Parameters
    ----------
    estimator : keras.models.Model
        Model whose layers are to be accessed.
        See :func:`explain_prediction_keras`.

    layer : int or str or keras.layers.Layer, optional
        Desired layer.
        See :func:`explain_prediction_keras`.

    Returns
    -------
    activation_layer : keras.layers.Layer
        A keras ``Layer`` instance that will be targetted.


    :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
    """        
    if layer is None:
        # Automatically get the layer if not provided
        activation_layer = search_layer_backwards(estimator, is_suitable_activation_layer)
    elif isinstance(layer, Layer):
        activation_layer = layer
    elif isinstance(layer, int):
        # bottom-up horizontal graph traversal
        activation_layer = estimator.get_layer(index=layer)
        # These can raise ValueError if the layer index / name specified is not found
    elif isinstance(layer, str):
        activation_layer = estimator.get_layer(name=layer)
    else:
        raise TypeError('Invalid layer (must be str, int, keras.layers.Layer, or None): %s' % layer)

    # TODO: validate activation_layer dimensions (is it possible to perform Grad-CAM on it?)
    return activation_layer


def search_layer_backwards(estimator, condition):
    # type: (Model, Callable[[Model, int], bool]) -> Layer
    """
    Search for a layer in ``estimator`` backwards (starting from the output layer),
    checking if the layer is suitable with the callable ``condition``,
    
    Parameters
    ----------
    estimator : keras.models.Model
        Model whose layers will be searched.
        See :func:`explain_prediction_keras`.

    condition : callable
        A callable that takes ``estimator`` and ``index`` arguments
        and returns a boolean.
        See :func:`is_suitable_activation_layer` as an example.
    
    Returns
    -------
    layer : keras.layers.Layer
        A suitable keras Layer instance.


    :raises ValueError: if suitable layer can not be found.
    """
    # we assume that this is a simple feedforward network
    # linear search in reverse
    i = len(estimator.layers)-1
    while i >= 0 and not condition(estimator, i):
        i -= 1
    if i >= 0:
        # linear search succeeded
        return estimator.get_layer(index=i)
    else:
        raise ValueError('Could not find a suitable target layer automatically.')


def is_suitable_activation_layer(estimator, i):
    # type: (Model, int) -> bool
    """
    Check whether
    the layer at index ``i`` matches what is required 
    by ``estimator``.
    
    Parameters
    ----------
    estimator : keras.models.Model
        Model from which to retrieve the layer.
        See :func:`explain_prediction_keras`.

    i : int
        Index into the ``estimator``'s layers list.

    Returns
    -------
    is_suitable : boolean
        whether the layer matches what is needed

    Notes
    -----

    Matching Criteria
        * Rank of the layer's output tensor.
    """
    # TODO: experiment with this, using many models and images, to find what works best
    # Some ideas: 
    # check layer type, i.e.: isinstance(l, keras.layers.Conv2D)
    # check layer name
    l = estimator.get_layer(index=i)
    # a check that asks 'can we resize this activation layer over the image?'
    rank = len(l.output_shape)
    required_rank = len(estimator.input_shape)
    return rank == required_rank


def grad_cam(estimator, doc, targets, activation_layer):
    # type: (Model, np.ndarray, Optional[list], Layer) -> Tuple[np.ndarray, int, float]
    """
    Generate a heatmap using Gradient-weighted Class Activation Mapping 
    (Grad-CAM) (https://arxiv.org/pdf/1610.02391.pdf).
    
    See :func:`explain_prediction_keras` for more information about the 
    ``estimator``, ``doc`` and ``targets`` parameters.

    Parameters
    ----------
    estimator : keras.models.Model
        Model to Grad-CAM on.

    doc : numpy.ndarray
        Input image to ``estimator``.

    targets : list or None
        Prediction to focus on that can be made by ``estimator``.

    activation_layer : keras.layers.Layer
        Hidden layer in ``estimator`` to differentiate with respect to.

    
    Returns
    -------
    (heatmap, predicted_idx, score) : (numpy.ndarray, int, float)
        A Grad-CAM localization map,
        the predicted class ID, and
        the score (i.e. probability) for that class.


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
    weights, activations, grads, predicted_idx, score = grad_cam_backend(estimator, doc, targets, activation_layer)

    # TODO: might want to replace np operations with keras backend operations

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
    return lmap, predicted_idx, score


def grad_cam_backend(estimator, # type: Model
    doc, # type: np.ndarray
    targets, # type: Optional[List[int]]
    activation_layer # type: Layer
    ):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]
    """
    Compute the terms required by the Grad-CAM formula and its by-products.
    
    See :func:`eli5.keras.grad_cam` for description of the 
    ``estimator``, ``doc``, ``targets``, and ``activation_layer``
    parameters.

    Returns
    -------
    (weights, activations, gradients, predicted_idx, score) : (numpy.ndarray, ..., int, float)
        Values of variables.
    """
    output = estimator.output
    predicted_idx = get_target_prediction(targets, output)
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


def get_target_prediction(targets, output):
    # type: (Union[None, list], K.variable) -> K.variable
    """
    Get a prediction ID (index into the final layer of the model), 
    using ``targets``.
    
    Parameters
    ----------
    targets : list, optional
        A list of predictions. Only length one is currently supported.
        If None, top prediction is made automatically by taking 
        the unit in the output layer with
        the largest score.
        See documentation of ``explain_prediction_keras``.

    output : K.variable
        Input tensor, rank 2.
        This will be searched for the maximum score and indexed.


    Returns
    -------
    predicted_idx : K.variable
        Prediction ID to focus on, as a suitable rank 1 Keras backend tensor.


    :raises ValueError: if targets is a list with more than one item.  
        *Currently only a single target prediction is supported*.
    :raises TypeError: if targets is not list or None.
    """
    # TODO: take in a single target as well, not just a list, 
    # consider changing signature / types for explain_prediction generic function
    # TODO: need to find a way to show the label for the passed prediction 
    # as well as its probability

    # FIXME: this is hard to test, as output must be evaluated first

    # TODO: maybe do the sum / loss in this function instead of grad_cam. Return a tensor.
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
        # print('Taking top prediction: %d' % predicted_idx)
        # TODO: append this to description / log instead of printing
    else:
        raise TypeError('Invalid argument "targets" (must be list or None): %s' % targets)
        # TODO: in the future, accept different ways to specify target
        # label (str), float (in regression tasks), int (not a list) etc.
    return predicted_idx


def image_from_path(img_path, image_shape=None):
    # type: (str, Optional[Tuple[int, int]]) -> np.ndarray
    """
    Load a single image from disk, with an optional resize.

    Parameters
    ----------
    img_path : str
        Path to a single image file.

    image_shape : tuple[int], optional
        A (height, width) tuple that indicates the dimensions that the 
        image is to be resized to.


    Returns
    -------
    doc : numpy.ndarray
        An array representing the image, 
        suitable as input to a model (batch axis included).
    """
    # TODO: Take in PIL image object, or an array
    # "pipeline": path str -> PIL image -> numpy array
    # TODO: multiple images
    im = keras.preprocessing.image.load_img(img_path, target_size=image_shape)
    x = keras.preprocessing.image.img_to_array(im)

    # we need to insert an axis at the 0th position to indicate the batch size (required by the model's input)
    x = np.expand_dims(x, axis=0)
    return x